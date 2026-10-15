"""Configured BenchMARL training functions for BenchMAX games."""

from __future__ import annotations

import contextlib
import copy
import io
import logging
import tempfile
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

from omegaconf import OmegaConf
from omegaconf.errors import ValidationError

from benchmax.core import Game
from benchmax.integrations.benchmarl.provider import algorithm_config_class
from benchmax.policy import PolicyProfile
from benchmax.providers import require_provider
from benchmax.seeding import derive_seed, seed_process
from benchmax.training.algorithm import _validate_training_controls
from benchmax.types import Player

if TYPE_CHECKING:
    from benchmarl.algorithms.common import AlgorithmConfig
    from benchmarl.experiment import Experiment, ExperimentConfig
    from benchmarl.experiment.callback import Callback
    from benchmarl.models import MlpConfig

    from benchmax.execution.specs import TrainingSpec


def _apply(defaults: Any, overrides: Mapping[str, Any] | None) -> Any:
    """Merge validated overrides onto one BenchMARL dataclass."""

    if not overrides:
        return defaults

    known = {field.name for field in fields(defaults)}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise ValueError(f"unknown {type(defaults).__name__} settings: {unknown}; available settings: {sorted(known)}")

    try:
        structured = OmegaConf.structured(defaults)
    except ValidationError:
        for key, value in overrides.items():
            setattr(defaults, key, value)

        return defaults

    return OmegaConf.to_object(OmegaConf.merge(structured, dict(overrides)))


@dataclass(frozen=True)
class BenchMARLTrainingFunction:
    """A configured callable that delegates its training loop to BenchMARL."""

    algorithm: AlgorithmConfig
    experiment: ExperimentConfig
    model: MlpConfig
    total_frames: int
    device: str = "auto"
    verbose: bool = True
    cuda_graphs: bool = True
    torch_threads: int | None = 1
    matmul_precision: str | None = None

    def __call__(self, game: Game, *, seed: int) -> PolicyProfile:
        return self._train(game, seed=seed)

    def best_response(
        self, game: Game, opponents: PolicyProfile, player: Player, *, seed: int,
        checkpoint_frames: tuple[int, ...] = (),
        on_checkpoint: Callable[[PolicyProfile, int], None] | None = None,
    ) -> PolicyProfile:
        """Train one player, optionally observing frozen policies after completed updates."""
        return self._train(
            game, seed=seed, response_player=player, opponents=opponents,
            checkpoint_frames=checkpoint_frames, on_checkpoint=on_checkpoint,
        )

    def _train(
        self, game: Game, *, seed: int, response_player: Player | None = None, opponents: PolicyProfile | None = None,
        checkpoint_frames: tuple[int, ...] = (),
        on_checkpoint: Callable[[PolicyProfile, int], None] | None = None,
    ) -> PolicyProfile:
        experiment_seed = derive_seed(seed, "training", "experiment")
        collector_seed = derive_seed(seed, "training", "collector")
        seed_process(experiment_seed)
        import torch

        torch.manual_seed(experiment_seed)
        train_device = _resolve_device(self.device)

        with (
            _training_threads(train_device, self.torch_threads),
            _training_matmul_precision(train_device, self.matmul_precision),
            tempfile.TemporaryDirectory(prefix="benchmax-training-") as output_dir,
        ):
            capture: contextlib.AbstractContextManager[object] = contextlib.nullcontext()

            if not self.verbose:
                stack = contextlib.ExitStack()
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                capture = stack

            torchrl_logger = logging.getLogger("torchrl")
            previous_log_level = torchrl_logger.level
            torchrl_logger.setLevel(logging.WARNING)

            try:
                with warnings.catch_warnings(), capture:
                    warnings.filterwarnings(
                        "ignore", message="SyncDataCollector has been deprecated", category=DeprecationWarning
                    )
                    warnings.filterwarnings(
                        "ignore", message="size_average and reduce args will be deprecated", category=UserWarning
                    )
                    checkpoints = None
                    if on_checkpoint is not None:
                        from benchmax.integrations.benchmarl.env import _ResponseCheckpoints

                        checkpoints = _ResponseCheckpoints(
                            game, response_player, checkpoint_frames or (self.total_frames,), on_checkpoint,
                        )
                    experiment = _create_experiment(
                        game, self, seed=experiment_seed, device=train_device, output_dir=output_dir,
                        response_player=response_player, opponents=opponents,
                        callbacks=[checkpoints] if checkpoints is not None else None,
                    )

                    try:
                        experiment.collector.set_seed(collector_seed)
                    except BaseException:
                        experiment.close()
                        raise

                    experiment.run()
                    if checkpoints is not None:
                        # A final collection batch may have no completed
                        # decision transitions and therefore no optimizer hook.
                        checkpoints._capture()
                    if response_player is None:
                        return _policy_from_experiment(experiment, game)
                    if checkpoints is not None and checkpoints.latest is not None:
                        return checkpoints.latest
                    from benchmax.integrations.benchmarl.env import _BenchMARLPolicyProfile

                    return PolicyProfile(_BenchMARLPolicyProfile(experiment, game, players=(response_player,)))
            finally:
                torchrl_logger.setLevel(previous_log_level)


def build_training_function(
    *, provider: str, name: str, total_frames: int = 100_000, device: str = "auto", verbose: bool = True,
    cuda_graphs: bool = True, torch_threads: int | None = 1, matmul_precision: str | None = None,
    params: Mapping[str, Any] | None = None
) -> BenchMARLTrainingFunction:
    """Build one configured BenchMARL training callable."""

    require_provider(provider)

    if provider != "benchmarl":
        raise ValueError(f"the {provider!r} provider does not supply training algorithms")

    if total_frames is None:
        raise ValueError("total_frames must be a positive integer")

    _validate_training_controls(total_frames=total_frames, device=device, verbose=verbose)

    if not isinstance(cuda_graphs, bool):
        raise TypeError("cuda_graphs must be a bool")

    if torch_threads is not None and (
        not isinstance(torch_threads, int) or isinstance(torch_threads, bool) or torch_threads <= 0
    ):
        raise ValueError("torch_threads must be a positive integer or None")

    if matmul_precision not in (None, "ieee", "tf32"):
        raise ValueError("matmul_precision must be 'ieee', 'tf32', or None")

    try:
        from benchmarl.experiment import ExperimentConfig
        from benchmarl.models import MlpConfig
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("BenchMARL training is unavailable; install benchmax[training]") from error

    settings = dict(params or {})
    algorithm = settings.get("algorithm")
    experiment = settings.get("experiment")
    model = settings.get("model")
    unknown = sorted(set(settings) - {"algorithm", "experiment", "model"})
    if unknown:
        raise ValueError(f"unknown BenchMARL parameter groups: {unknown}; expected algorithm, experiment, or model")

    algorithm_class = algorithm_config_class(name)
    return BenchMARLTrainingFunction(
        algorithm=_apply(algorithm_class.get_from_yaml(), algorithm),
        experiment=_apply(ExperimentConfig.get_from_yaml(), experiment), model=_apply(MlpConfig.get_from_yaml(), model),
        total_frames=total_frames, device=device, verbose=verbose, cuda_graphs=cuda_graphs, torch_threads=torch_threads,
        matmul_precision=matmul_precision
    )


def load_benchmarl_training_function(spec: TrainingSpec) -> BenchMARLTrainingFunction:
    """Resolve one registered algorithm and require the BenchMARL implementation."""

    from benchmax.training.algorithm import load_training_function

    training_function = load_training_function(
        spec.algorithm.name, total_frames=spec.total_frames, **spec.algorithm.overrides
    )
    if not isinstance(training_function, BenchMARLTrainingFunction):
        raise TypeError(f"algorithm {spec.algorithm.name!r} is not backed by BenchMARL")

    return training_function


def _resolve_device(device: str) -> str:
    """Pick a concrete Torch device, honouring an explicit choice."""

    if device != "auto":
        return device

    import torch

    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


@contextlib.contextmanager
def _training_threads(device: str, threads: int | None):
    """Avoid oversubscribing the small CPU rollout networks during CUDA training."""
    import torch

    previous = torch.get_num_threads()
    change = torch.device(device).type == "cuda" and threads is not None and threads != previous
    if change:
        torch.set_num_threads(threads)

    try:
        yield
    finally:
        if change:
            torch.set_num_threads(previous)


@contextlib.contextmanager
def _training_matmul_precision(device: str, precision: str | None):
    """Opt into TF32 for CUDA matrix products, restoring the caller's setting."""
    import torch

    if torch.device(device).type != "cuda" or precision is None:
        yield
        return

    previous = torch.backends.cuda.matmul.fp32_precision
    torch.backends.cuda.matmul.fp32_precision = precision

    try:
        yield
    finally:
        torch.backends.cuda.matmul.fp32_precision = previous


def _create_experiment(
    game: Game, training_function: BenchMARLTrainingFunction, *, seed: int, device: str, output_dir: str,
    response_player: Player | None = None, opponents: PolicyProfile | None = None,
    callbacks: list[Callback] | None = None,
) -> Experiment:
    """Construct a fresh experiment from one reusable configured callable."""

    from benchmarl.experiment import Experiment

    from benchmax.integrations.benchmarl.env import _BenchmaxTask
    from benchmax.integrations.benchmarl.ppo import _can_use_cuda_graphs, _PPOExperiment

    algorithm, experiment_config, model = copy.deepcopy(
        (training_function.algorithm, training_function.experiment, training_function.model)
    )
    experiment_config.sampling_device = "cpu"
    experiment_config.train_device = device
    experiment_config.buffer_device = device
    experiment_config.prefer_continuous_actions = False
    experiment_config.max_n_frames = training_function.total_frames
    experiment_config.max_n_iters = None
    if not algorithm.on_policy():
        batch_frames = experiment_config.off_policy_collected_frames_per_batch
        num_envs = experiment_config.off_policy_n_envs_per_worker
        batch_frames = -(-batch_frames // num_envs) * num_envs
        collected_frames = -(-training_function.total_frames // batch_frames) * batch_frames
        experiment_config.off_policy_memory_size = min(experiment_config.off_policy_memory_size, collected_frames)
    experiment_config.evaluation = False
    experiment_config.render = False
    experiment_config.loggers = []
    experiment_config.create_json = False
    experiment_config.checkpoint_interval = 0
    experiment_config.checkpoint_at_end = False
    experiment_config.save_folder = output_dir

    if response_player is not None:
        # Response training must optimize the same return as evaluation. Each
        # collector step remains one original game step, including opponent turns.
        experiment_config.gamma = 1.0 if game.discount is None else game.discount

    task = _BenchmaxTask(
        game, max_episode_steps=game.max_episode_steps, use_state=algorithm.has_centralized_critic(),
        response_player=response_player, opponents=opponents,
    )
    experiment_class = Experiment
    from benchmarl.algorithms import IqlConfig

    if type(algorithm) is IqlConfig:
        from benchmax.integrations.benchmarl.iql import _IqlConfig, _IQLExperiment

        algorithm = _IqlConfig(delay_value=algorithm.delay_value, loss_function=algorithm.loss_function)
        experiment_class = _IQLExperiment
    elif training_function.cuda_graphs and _can_use_cuda_graphs(algorithm, experiment_config, model):
        experiment_class = _PPOExperiment

    return experiment_class(
        task=task, algorithm_config=algorithm, model_config=model, seed=seed, config=experiment_config,
        callbacks=callbacks,
    )


def _policy_from_experiment(experiment: Experiment, game: Game) -> PolicyProfile:
    from benchmax.integrations.benchmarl.env import _BenchMARLPolicyProfile

    return PolicyProfile(_BenchMARLPolicyProfile(experiment, game))
