"""BenchMARL training adapter for BenchMAX games."""

from __future__ import annotations

import copy
import functools
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from benchmarl.environments import TaskClass
from benchmarl.experiment import Experiment
from benchmarl.experiment.callback import Callback
from tensordict import TensorDict
from torchrl.data import Binary, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from benchmax._rollout import sample_actions
from benchmax.core import ActionSpace, Game, Player, TimeStep
from benchmax.execution.artifacts.interface import _PolicyInterface
from benchmax.policy import PolicyProfile
from benchmax.seeding import derive_seed, python_generator


def _group_name(player: Player) -> str:
    return f"player_{player}"


def _make_scalar_specs(
    game: Game, *, device: torch.device, use_state: bool, players: tuple[Player, ...] | None = None
) -> tuple[Composite, Composite, Composite, Composite]:
    observation_specs: dict[str, Any] = {}
    action_specs: dict[str, Composite] = {}
    reward_specs: dict[str, Composite] = {}

    for player in game.players if players is None else players:
        group = _group_name(player)
        observation_size = math.prod(game.observation_spaces[player])
        action_count = len(game.action_spaces[player])
        group_shape = torch.Size((1,))
        observation_specs[group] = Composite(
            {
                "observation": Unbounded(shape=(1, observation_size + 1), dtype=torch.float32, device=device),
                "action_mask": Binary(n=action_count, shape=(1, action_count), dtype=torch.bool, device=device),
            },
            shape=group_shape,
            device=device,
        )
        action_specs[group] = Composite(
            {
                "action": Categorical(n=action_count, shape=group_shape, dtype=torch.int64, device=device)
            },
            shape=group_shape,
            device=device,
        )
        reward_specs[group] = Composite(
            {
                "reward": Unbounded(shape=(1, 1), dtype=torch.float32, device=device)
            },
            shape=group_shape,
            device=device,
        )

    if use_state:
        assert game.state_space is not None
        observation_specs["state"] = Unbounded(shape=(math.prod(game.state_space),), dtype=torch.float32, device=device)

    return (
        Composite(observation_specs, shape=(), device=device),
        Composite(action_specs, shape=(), device=device),
        Composite(reward_specs, shape=(), device=device),
        Composite(
            {
                key: Categorical(n=2, shape=(1,), dtype=torch.bool, device=device)
                for key in ("done", "terminated", "truncated")
            },
            shape=(),
            device=device,
        ),
    )


class _BatchedBenchmaxEnv(EnvBase):
    """Expose a batch of BenchMAX environments through TorchRL."""

    def __init__(
        self, game: Game, *, num_envs: int, seed: int | None, device: torch.device | str, use_state: bool = False,
        response_player: Player | None = None, opponents: PolicyProfile | None = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")

        super().__init__(device=device, batch_size=torch.Size((num_envs,)))
        self._game = game
        self._players = game.players if response_player is None else (response_player,)
        self._opponents = opponents
        self._opponent_players = tuple(p for p in game.players if p not in self._players)
        self._opponent_rngs = []
        self._num_envs = num_envs
        self._envs = [self._game.new_env() for _ in range(num_envs)]
        self._time_steps: list[TimeStep | None] = [None] * num_envs
        self._pending_seeds: list[int | None] = [None] * num_envs
        self._use_state = use_state
        if use_state and self._game.state_space is None:
            raise ValueError(f"{game.name!r} does not expose global training state")

        self._groups = {
            player: _group_name(player) for player in self._players
        }
        self._observation_sizes = {
            player: math.prod(self._game.observation_spaces[player])
            for player in self._players
        }
        self._action_counts = {
            player: len(self._game.action_spaces[player])
            for player in self._players
        }
        self._state_size = (
            math.prod(self._game.state_space)
            if self._game.state_space is not None
            else None
        )
        self.group_map = {group: [group] for group in self._groups.values()}
        self._inactive_observations = {
            player: (0.0,) * self._observation_sizes[player] + (0.0,)
            for player in self._players
        }
        self._inactive_masks = {
            player: tuple(
                index == 0 for index in range(self._action_counts[player])
            )
            for player in self._players
        }

        scalar_specs = _make_scalar_specs(
            self._game, device=self.device, use_state=self._use_state, players=self._players,
        )
        self.observation_spec = scalar_specs[0].expand(num_envs)
        self.action_spec = scalar_specs[1].expand(num_envs)
        self.reward_spec = scalar_specs[2].expand(num_envs)
        self.done_spec = scalar_specs[3].expand(num_envs)
        self._set_seed(seed)

    def _set_seed(self, seed: int | None) -> None:
        if seed is None:
            self._pending_seeds = [None] * self._num_envs
            return

        self._pending_seeds = [derive_seed(seed, "benchmarl-environment", row) for row in range(self._num_envs)]
        if self._opponents is not None:
            self._opponent_rngs = [
                python_generator(derive_seed(seed, "benchmarl-opponents", row)) for row in range(self._num_envs)
            ]

    def rand_action(self, tensordict: TensorDict | None = None) -> TensorDict:
        """Make collector warm-up actions respect the current legal moves."""
        if tensordict is None:
            tensordict = self._encode_batch(include_rewards=False)
        for group in self._groups.values():
            mask = tensordict[group, "action_mask"].reshape(self._num_envs, -1)
            tensordict[group, "action"] = torch.multinomial(mask.float(), 1)
        return tensordict

    def _reset(self, tensordict: TensorDict | None = None) -> TensorDict:
        reset_mask = self._row_mask(tensordict, "_reset", default=True)

        for row in reset_mask.nonzero(as_tuple=True)[0].tolist():
            self._time_steps[row] = self._envs[row].reset(
                seed=self._pending_seeds[row]
            )
            self._pending_seeds[row] = None

        return self._encode_batch(include_rewards=False)

    def _step(self, tensordict: TensorDict) -> TensorDict:
        if any(time_step is None for time_step in self._time_steps):
            raise RuntimeError("reset must be called before step")

        step_mask = self._row_mask(tensordict, "_step", default=True)
        action_columns = {
            player: tensordict[self._groups[player], "action"]
            .reshape(self._num_envs)
            .tolist()
            for player in self._players
        }

        rows = step_mask.nonzero(as_tuple=True)[0].tolist()
        opponent_actions = (
            sample_actions(
                self._game, self._opponents, [self._time_steps[row] for row in rows],
                [self._opponent_rngs[row] for row in rows], players=self._opponent_players,
            ) if self._opponents is not None else [{} for _ in rows]
        )

        for row, actions in zip(rows, opponent_actions, strict=True):
            time_step = self._time_steps[row]
            assert time_step is not None

            for player in time_step.players_to_act:
                if player not in self._players:
                    continue
                action_index = int(action_columns[player][row])
                action_space = self._game.action_spaces[player]
                if not 0 <= action_index < len(action_space):
                    raise ValueError(
                        f"invalid categorical action {action_index} "
                        f"for player {player}"
                    )

                actions[player] = action_space[action_index]

            self._time_steps[row] = self._envs[row].step(actions)

        return self._encode_batch(include_rewards=True, transition_mask=step_mask)

    def _encode_batch(self, *, include_rewards: bool, transition_mask: torch.Tensor | None = None) -> TensorDict:
        if any(time_step is None for time_step in self._time_steps):
            raise RuntimeError("every environment must be reset before encoding")

        time_steps = [
            time_step for time_step in self._time_steps if time_step is not None
        ]
        stepped = (
            [True] * self._num_envs
            if transition_mask is None
            else transition_mask.tolist()
        )
        fields: dict[str, Any] = {}

        for player in self._players:
            group = self._groups[player]
            observation_size = self._observation_sizes[player]
            observation_rows: list[tuple[float, ...]] = []
            mask_rows: list[tuple[bool, ...]] = []

            for time_step in time_steps:
                if player in time_step.players_to_act:
                    observation = time_step.observations[player]
                    if len(observation) != observation_size:
                        raise ValueError(
                            f"observation for player {player} has length "
                            f"{len(observation)}; expected {observation_size}"
                        )

                    observation_rows.append((*observation, 1.0))
                    legal = set(time_step.legal_actions[player])
                    mask_rows.append(
                        tuple(
                            action in legal
                            for action in self._game.action_spaces[player]
                        )
                    )
                else:
                    observation_rows.append(self._inactive_observations[player])
                    mask_rows.append(self._inactive_masks[player])

            group_fields: dict[str, torch.Tensor] = {
                "observation": torch.tensor(observation_rows, dtype=torch.float32, device=self.device).unsqueeze(1),
                "action_mask": torch.tensor(mask_rows, dtype=torch.bool, device=self.device).unsqueeze(1),
            }

            if include_rewards:
                group_fields["reward"] = torch.tensor(
                    [
                        time_step.rewards[player] if stepped[row] else 0.0
                        for row, time_step in enumerate(time_steps)
                    ],
                    dtype=torch.float32,
                    device=self.device,
                ).reshape(self._num_envs, 1, 1)

            fields[group] = TensorDict(group_fields, batch_size=(*self.batch_size, 1), device=self.device)

        fields.update(
            {
                "done": torch.tensor(
                    [
                        time_step.terminated or time_step.truncated
                        for time_step in time_steps
                    ],
                    dtype=torch.bool,
                    device=self.device,
                ).unsqueeze(-1),
                "terminated": torch.tensor(
                    [time_step.terminated for time_step in time_steps], dtype=torch.bool, device=self.device
                ).unsqueeze(-1),
                "truncated": torch.tensor(
                    [time_step.truncated for time_step in time_steps], dtype=torch.bool, device=self.device
                ).unsqueeze(-1),
            }
        )

        if self._use_state:
            assert self._state_size is not None
            states = [env.state() for env in self._envs]

            for state in states:
                if len(state) != self._state_size:
                    raise ValueError(
                        f"global state has length {len(state)}; "
                        f"expected {self._state_size}"
                    )

            fields["state"] = torch.tensor(states, dtype=torch.float32, device=self.device)

        return TensorDict(fields, batch_size=self.batch_size, device=self.device)

    def _row_mask(self, tensordict: TensorDict | None, key: str, *, default: bool) -> torch.Tensor:
        if tensordict is None:
            return torch.full((self._num_envs,), default, dtype=torch.bool, device=self.device)

        mask = tensordict.get(key, None)
        if mask is None:
            return torch.full((self._num_envs,), default, dtype=torch.bool, device=self.device)

        return mask.reshape(self._num_envs, -1).any(-1)


class _BenchmaxTask(TaskClass):
    """One generic BenchMARL task for any BenchMAX game."""

    def __init__(
        self, game: Game, *, max_episode_steps: int, use_state: bool = False,
        response_player: Player | None = None, opponents: PolicyProfile | None = None,
    ) -> None:
        super().__init__(name=game.name, config={"max_episode_steps": max_episode_steps})
        self._game = game
        self._use_state = use_state
        self._response_player = response_player
        self._opponents = opponents

    def __getstate__(self) -> dict[str, Any]:
        # BenchMARL always writes config.pkl, even with checkpoints disabled.
        # This temporary metadata is not BenchMAX's persistence format. Keep
        # live policy oracles (which may own locks/devices) in memory only.
        state = self.__dict__.copy()
        if self._response_player is not None:
            state["_opponents"] = None
            game = state["_game"] = copy.copy(self._game)
            game.evaluator = None
            game._benchmax_configured_evaluator = None
        return state

    def get_env_fun(
        self, num_envs: int, continuous_actions: bool, seed: int | None, device: torch.device | str
    ) -> Callable[[], EnvBase]:
        del continuous_actions
        return functools.partial(
            _BatchedBenchmaxEnv, self._game, num_envs=num_envs, seed=seed, device=device, use_state=self._use_state,
            response_player=self._response_player, opponents=self._opponents,
        )

    def supports_continuous_actions(self) -> bool:
        return False

    def get_env_transforms(self, env: EnvBase) -> list:
        if self._response_player is not None and self._game.discount is not None:
            from torchrl.envs import StepCounter

            # Reset continuing trajectories without labeling the cutoff as a
            # terminal state: the critic must bootstrap the discounted tail.
            return [StepCounter(max_steps=self.max_steps(env))]
        return []

    def supports_discrete_actions(self) -> bool:
        return True

    def max_steps(self, env: EnvBase) -> int:
        del env
        return int(self.config["max_episode_steps"])

    def has_render(self, env: EnvBase) -> bool:
        del env
        return False

    def group_map(self, env: EnvBase) -> dict[str, list[str]]:
        return dict(env.group_map)

    def observation_spec(self, env: EnvBase) -> Composite:
        return self._group_spec(env, drop="action_mask")

    def info_spec(self, env: EnvBase) -> None:
        del env

    def state_spec(self, env: EnvBase) -> Composite | None:
        if not self._use_state:
            return None

        return Composite({"state": env.full_observation_spec_unbatched["state"].clone()}, shape=(), device=env.device)

    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec_unbatched.clone()

    def action_mask_spec(self, env: EnvBase) -> Composite:
        return self._group_spec(env, drop="observation")

    def _group_spec(self, env: EnvBase, *, drop: str) -> Composite:
        """Per-group observation entries without the shared state or the ``drop`` key."""
        spec = env.full_observation_spec_unbatched.clone()

        if self._use_state:
            del spec["state"]

        for group in self.group_map(env):
            del spec[group, drop]

        return spec

    @staticmethod
    def env_name() -> str:
        return "benchmax"


class _ResponseCheckpoints(Callback):
    """Observe occasional completed updates without changing PPO minibatches."""

    def __init__(self, game, player, frames, select):
        super().__init__()
        self.game, self.player, self.frames, self.select = game, player, frames, select
        self.next_checkpoint = 0
        self.latest = None

    def __getstate__(self):
        # BenchMARL writes temporary config.pkl even without persistence. The
        # selection closure and its live policies belong only to this run.
        return {**self.__dict__, "experiment": None, "game": None, "select": None, "latest": None}

    def on_train_end(self, training_td, group):
        # BenchMARL calls this hook after each group's optimizer updates.
        if self.player is None and group != next(reversed(self.experiment.train_group_map)):
            return
        self._capture()

    def _capture(self):
        frames = self.experiment.total_frames
        if self.next_checkpoint == len(self.frames) or frames < self.frames[self.next_checkpoint]:
            return
        # Several thresholds may fall within one collection batch. Validate the
        # resulting policy once, after all of that batch's optimizer updates.
        while self.next_checkpoint < len(self.frames) and frames >= self.frames[self.next_checkpoint]:
            self.next_checkpoint += 1
        players = None if self.player is None else (self.player,)
        self.latest = PolicyProfile(_BenchMARLPolicyProfile(self.experiment, self.game, players=players))
        self.select(self.latest, frames)


class _BenchMARLPolicyProfile:
    """Frozen batched policy profile backed by trained Torch modules."""

    def __init__(self, experiment: Experiment, game: Game, *, players: tuple[Player, ...] | None = None) -> None:
        self._players = tuple(game.players) if players is None else players
        self._observation_shapes = {p: game.observation_spaces[p] for p in self._players}
        self._observation_sizes = {
            player: math.prod(game.observation_spaces[player])
            for player in self._players
        }
        self._actions = {p: game.action_spaces[p] for p in self._players}
        self._interface = _PolicyInterface.from_game(game, self._players)
        self._policies = {
            player: copy.deepcopy(experiment.group_policies[_group_name(player)]).to("cpu").eval()
            for player in self._players
        }

    def _state_dicts(self) -> dict[Player, dict[str, Any]]:
        """Return CPU state for BenchMAX's safe artifact writer.

        TensorDict modules include small structural extra-state values such as
        ``torch.Size`` and ``None`` alongside ordinary Torch tensors. The
        artifact layer encodes those explicitly as JSON rather than pickling
        the state dictionary.
        """

        return {
            player: {
                name: value.detach().cpu()
                if isinstance(value, torch.Tensor)
                else copy.deepcopy(value)
                for name, value in policy.state_dict().items()
            }
            for player, policy in self._policies.items()
        }

    def __call__(
        self, player: Player, observations: np.ndarray, action_space: ActionSpace, legal_action_mask: np.ndarray, /
    ) -> np.ndarray:
        if player not in self._policies:
            raise ValueError(f"unknown player {player}")

        observation_values = np.asarray(observations)
        observation_shape = self._observation_shapes[player]
        if observation_values.ndim != len(observation_shape) + 1:
            raise ValueError(
                f"observations for player {player} have shape "
                f"{observation_values.shape!r}; expected "
                f"(batch_size, {', '.join(map(str, observation_shape))})"
            )

        batch_size = observation_values.shape[0]
        expected_observation_shape = (batch_size, *observation_shape)
        if observation_values.shape != expected_observation_shape:
            raise ValueError(
                f"observations for player {player} have shape "
                f"{observation_values.shape!r}; expected "
                f"{expected_observation_shape!r}"
            )

        actions = self._actions[player]
        if tuple(action_space) != actions:
            raise ValueError(
                f"action space for player {player} is {tuple(action_space)!r}; "
                f"expected {actions!r}"
            )

        mask = np.asarray(legal_action_mask)
        expected_mask_shape = (batch_size, len(actions))
        if mask.shape != expected_mask_shape or not np.issubdtype(mask.dtype, np.bool_):
            raise ValueError(
                f"legal-action mask for player {player} must be a boolean array "
                f"with shape {expected_mask_shape!r}"
            )

        group = _group_name(player)
        observation_tensor = torch.as_tensor(
            observation_values, dtype=torch.float32
        ).reshape(batch_size, self._observation_sizes[player])
        encoded_observations = torch.cat(
            (observation_tensor, torch.ones((batch_size, 1), dtype=torch.float32)), dim=-1
        ).unsqueeze(1)
        action_mask = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(1)
        tensordict = TensorDict(
            {
                group: TensorDict(
                    {
                        "observation": encoded_observations,
                        "action_mask": action_mask,
                    },
                    batch_size=(batch_size, 1),
                )
            },
            batch_size=(batch_size,),
        )

        with torch.inference_mode():
            policy = self._policies[player]
            if hasattr(policy, "get_dist"):
                probabilities = policy.get_dist(tensordict).probs
            else:
                from torchrl.envs import ExplorationType, set_exploration_type

                with set_exploration_type(ExplorationType.DETERMINISTIC):
                    action = policy(tensordict)[group, "action"]
                probabilities = torch.nn.functional.one_hot(action, len(actions)).to(torch.float32)

        expected_values = batch_size * len(actions)
        if probabilities.numel() != expected_values:
            raise RuntimeError(
                f"trained policy for player {player} returned probability shape "
                f"{tuple(probabilities.shape)!r}; expected {expected_values} values"
            )

        return probabilities.reshape(batch_size, len(actions)).detach().cpu().numpy()
