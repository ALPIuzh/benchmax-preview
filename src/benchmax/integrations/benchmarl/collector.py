"""Buffered CPU collection for the adapter's single-agent PPO groups.

TorchRL still owns initialization, seeding, policy synchronization, trajectory
IDs, iteration and shutdown. The fixed BenchMAX observation layout lets us fill
one rollout allocation directly instead of building TensorDicts at every step.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from torchrl.collectors import Collector
from torchrl.envs import RewardSum

from benchmax._rollout import sample_actions
from benchmax.integrations.benchmarl.env import _BatchedBenchmaxEnv


class _Inference(torch.nn.Module):
    """Trace masked categorical inference, preserving its multinomial RNG draw."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, observation, mask):
        logits = self.net(observation)
        log_probs = logits.masked_fill(~mask, float("-inf")).log_softmax(-1)
        # Normalize once for both sampling and the selected action's log-prob.
        # Every adapter mask contains at least one legal action.
        action = torch.multinomial(log_probs.exp().reshape(-1, logits.shape[-1]), 1, True)
        action = action.reshape(logits.shape[:-1])
        return logits, action, log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)


class _PPOCollector(Collector):
    """Specialize only the CPU, feed-forward collector constructed by our adapter."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._base = self.env.base_env
        # A future task transform must keep its upstream behavior automatically.
        self._buffered = (
            type(self._base) is _BatchedBenchmaxEnv
            and self.env.device.type == "cpu"
            and len(self.env.transform) == 1
            and type(self.env.transform[0]) is RewardSum
        )
        if not self._buffered:
            return
        self._players = tuple(self._base._players)
        self._groups = self._base._groups
        self._actions = self._base._game.action_spaces
        self._action_indices = {
            player: {action: i for i, action in enumerate(self._actions[player])} for player in self._players
        }
        self._networks = []
        for actor in self.policy:
            model = actor[0]
            multi = model.mlp
            net = multi.get_stateful_net()
            if not multi.share_params:
                # There is exactly one agent per player group. These parameter
                # views share storage with the collector's CPU policy, so the
                # standard update_policy_weights_ also updates this network.
                for path, value in multi.params[0].flatten_keys(".").items():
                    parent, _, name = path.rpartition(".")
                    module = net.get_submodule(parent)
                    delattr(module, name)
                    module.register_parameter(name, torch.nn.Parameter(value, requires_grad=False))
            group = model.agent_group
            # Keep the parameters as shared views (do not freeze the trace).
            # Tracing must not advance the rollout's sampling RNG. Shapes are
            # fixed, but masks and legal actions remain runtime tensor inputs.
            with torch.random.fork_rng(devices=[]), warnings.catch_warnings():
                warnings.simplefilter("ignore", torch.jit.TracerWarning)
                inference = torch.jit.trace(
                    _Inference(net),
                    (self._carrier[group, "observation"], self._carrier[group, "action_mask"]),
                    check_trace=False,
                )
            self._networks.append((group, inference))

    def _encode_row(self, arrays, row, step, *, reward=False):
        for player in self._players:
            group = self._groups[player]
            observation = arrays[group, "observation"][row, 0]
            mask = arrays[group, "action_mask"][row, 0]
            observation.fill(0)
            mask.fill(False)
            if player in step.players_to_act:
                values = step.observations[player]
                if len(values) != len(observation) - 1:
                    raise ValueError(f"observation for player {player} has unexpected length {len(values)}")
                observation[:-1] = values
                observation[-1] = 1
                for action in step.legal_actions[player]:
                    index = self._action_indices[player].get(action)
                    if index is not None:
                        mask[index] = True
            else:
                mask[0] = True
            if reward:
                arrays[group, "reward"][row, 0, 0] = step.rewards.get(player, 0.0)
        arrays["done"][row, 0] = step.terminated or step.truncated
        arrays["terminated"][row, 0] = step.terminated
        arrays["truncated"][row, 0] = step.truncated
        if self._base._use_state:
            state = self._base._envs[row].state()
            if len(state) != arrays["state"].shape[-1]:
                raise ValueError(f"global state has unexpected length {len(state)}")
            arrays["state"][row] = state

    @torch.no_grad()
    def rollout(self):
        if not self._buffered:
            return super().rollout()
        base = self._base
        # Time-major storage makes each inference batch contiguous. The returned
        # TensorDict retains TorchRL's [environment, time] layout and time name.
        data = self._final_rollout.transpose(0, 1).contiguous()
        current = {
            key: value for key, value in data.items(True, True) if not isinstance(key, tuple) or key[0] != "next"
        }
        following = {
            key[1:] if len(key) > 2 else key[1]: value
            for key, value in data.items(True, True)
            if isinstance(key, tuple) and key[0] == "next"
        }
        for key, value in self._carrier.items(True, True):
            if key in current:
                current[key][0].copy_(value)
        arrays = {key: value.numpy() for key, value in current.items()}
        next_arrays = {key: value.numpy() for key, value in following.items()}
        for t in range(self.frames_per_batch):
            action_columns = {}
            for group, net in self._networks:
                logits, action, log_prob = net(current[group, "observation"][t], current[group, "action_mask"][t])
                current[group, "logits"][t].copy_(logits)
                current[group, "action"][t].copy_(action)
                current[group, "log_prob"][t].copy_(log_prob)
                action_columns[group] = action[:, 0].tolist()
            nxt = {key: value[t] for key, value in next_arrays.items()}
            opponent_actions = (
                sample_actions(
                    base._game, base._opponents, base._time_steps, base._opponent_rngs,
                    players=base._opponent_players,
                ) if base._opponents is not None else [{} for _ in base._time_steps]
            )
            for row, step in enumerate(base._time_steps):
                actions = opponent_actions[row]
                actions.update({
                    player: self._actions[player][action_columns[self._groups[player]][row]]
                    for player in step.players_to_act if player in self._groups
                })
                step = base._envs[row].step(actions)
                base._time_steps[row] = step
                self._encode_row(nxt, row, step, reward=True)
            for group in self._groups.values():
                nxt[group, "episode_reward"][:] = arrays[group, "episode_reward"][t] + nxt[group, "reward"]
            if t + 1 < self.frames_per_batch:
                carry = {key: value[t + 1] for key, value in arrays.items()}
                for key, value in nxt.items():
                    if key in carry:
                        carry[key][:] = value
                carry["collector", "traj_ids"][:] = arrays["collector", "traj_ids"][t]
            else:
                # A separate snapshot survives across rollout allocations and
                # keeps terminal next observations intact when rows reset.
                last = data[t].exclude("next").clone()
                for key, value in following.items():
                    if key in current:
                        last[key] = value[t].clone()
                carry = {key: value.numpy() for key, value in last.items(True, True)}
            rows = np.flatnonzero(nxt["done"][:, 0])
            if len(rows):
                carry["collector", "traj_ids"][rows] = self._traj_pool.get_traj_and_increment(
                    len(rows), device="cpu"
                ).numpy()
            for row in rows:
                step = base._envs[row].reset(seed=base._pending_seeds[row])
                base._pending_seeds[row] = None
                base._time_steps[row] = step
                self._encode_row(carry, row, step)
                for group in self._groups.values():
                    carry[group, "episode_reward"][row] = 0
        self._carrier = last
        return data.transpose(0, 1)
