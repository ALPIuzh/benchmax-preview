"""BenchMARL IQL with replay transitions between each player's decisions."""

from __future__ import annotations

import torch
from benchmarl.algorithms import Iql, IqlConfig
from benchmarl.experiment import Experiment
from tensordict import TensorDict


class _DecisionTransitions:
    """Keep unfinished decisions across collection batches, without dummy turns.

    A replay transition spans one action through the next decision of the same
    player (or the episode end). Rewards retain their original time discounts,
    and TorchRL's TD0 estimator uses ``steps_to_next_obs`` for the bootstrap discount.
    Collection and exploration budgets still count original environment frames.
    """

    def __init__(self, gamma: float):
        self.gamma = gamma
        self.pending = None

    def __call__(self, batch: TensorDict, group: str) -> TensorDict | None:
        current = batch[group]
        following = batch["next", group]
        if current["observation"][..., -1].bool().all() and (
            following["observation"][..., -1].bool() | batch["next", "done"]
        ).all():
            # Simultaneous games already contain decision-to-decision samples.
            next_group = following.select("observation", "action_mask", "reward").clone(False)
            next_group["done"] = batch["next", "done"].unsqueeze(-1)
            next_group["terminated"] = batch["next", "terminated"].unsqueeze(-1)
            return TensorDict({
                group: current.select("observation", "action_mask", "action"),
                "next": TensorDict({group: next_group}, batch_size=batch.batch_size),
                "steps_to_next_obs": torch.ones_like(next_group["reward"], dtype=torch.int64),
            }, batch_size=batch.batch_size).reshape(-1)
        n_envs, n_steps = batch.batch_size
        if self.pending is None:
            start = current[:, 0].select("observation", "action_mask", "action").clone()
            rewards = torch.zeros((n_envs, 1, 1), dtype=following["reward"].dtype)
            duration = torch.zeros((n_envs, 1, 1), dtype=torch.int64)
            active = torch.zeros(n_envs, dtype=torch.bool)
            self.pending = start, rewards, duration, active
        start, rewards, duration, active = self.pending
        completed = []
        for t in range(n_steps):
            acting = current["observation"][:, t, 0, -1].bool()
            for key in ("observation", "action_mask", "action"):
                start[key][acting] = current[key][:, t][acting]
            rewards[acting] = 0
            duration[acting] = 0
            active |= acting
            rewards[active] += self.gamma ** duration[active] * following["reward"][:, t][active]
            duration[active] += 1
            done = batch["next", "done"][:, t].squeeze(-1)
            finished = active & (following["observation"][:, t, 0, -1].bool() | done)
            if not finished.any():
                continue
            next_group = following[:, t][finished].select("observation", "action_mask").clone()
            next_group["reward"] = rewards[finished].clone()
            next_group["done"] = batch["next", "done"][:, t][finished].unsqueeze(-1)
            next_group["terminated"] = batch["next", "terminated"][:, t][finished].unsqueeze(-1)
            size = int(finished.sum())
            completed.append(TensorDict({
                group: start[finished].clone(),
                "next": TensorDict({group: next_group}, batch_size=(size,)),
                "steps_to_next_obs": duration[finished].clone(),
            }, batch_size=(size,)))
            active[finished] = False
        return torch.cat(completed) if completed else None


class _Iql(Iql):
    def process_batch(self, group, batch):
        # The experiment prepares these on the collection device, before the
        # upstream loop transfers its raw batch to the training device.
        del batch
        return self.decision_batches.pop(group).to(self.device)


class _IqlConfig(IqlConfig):
    @staticmethod
    def associated_class():
        return _Iql


class _IQLExperiment(Experiment):
    def _setup_algorithm(self):
        super()._setup_algorithm()
        self._decisions = {group: _DecisionTransitions(self.config.gamma) for group in self.group_map}

    def _on_batch_collected(self, batch):
        super()._on_batch_collected(batch)
        transitions = {group: convert(batch, group) for group, convert in self._decisions.items()}
        self.algorithm.decision_batches = {group: td for group, td in transitions.items() if td is not None}
        self._raw_batch_keys = tuple(batch.keys())
        # Upstream unconditionally samples replay for every training group.
        # Skip groups without completed decisions, including an empty first
        # batch, while advancing their exploration clock by collected frames.
        self.train_group_map = {
            group: agents for group, agents in self.group_map.items() if transitions[group] is not None
        }
        for group in self.group_map.keys() - self.train_group_map.keys():
            self.group_policies[group][-1].step(batch.numel())

    def _get_excluded_keys(self, group):
        # process_batch supplies the prepared transitions. Avoid copying the
        # discarded raw rollout to CUDA first in the upstream training loop.
        return self._raw_batch_keys
