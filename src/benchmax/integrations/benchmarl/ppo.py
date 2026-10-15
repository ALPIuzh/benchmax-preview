"""CUDA graph execution of BenchMARL's existing discrete PPO updates.

Only the repeated minibatch computation is captured. GAE, replay sampling, and
callbacks still run in BenchMARL. Fixed replay storage can be indexed inside the
graph. No extra optimizer steps are performed for warmup or capture.
"""

from __future__ import annotations

import gc
from collections.abc import Callable

import torch
from benchmarl.algorithms import IppoConfig, MappoConfig
from benchmarl.experiment import Experiment
from tensordict import TensorDict, TensorDictBase
from torchrl.data import LazyTensorStorage, SamplerWithoutReplacement, TensorDictReplayBuffer
from torchrl.envs import SelectTransform


def _categorical_entropy(dist, adv_shape):
    # Our adapter supplies MaskedCategorical, whose analytic entropy is finite
    # for finite logits and a nonempty legal-action mask. TorchRL's generic
    # Monte Carlo fallback checks this with a GPU-to-CPU synchronization.
    return dist.entropy().unsqueeze(-1)


class _CudaGraphUpdate:
    """Capture a fixed-shape update after three real minibatches on a side stream."""

    def __init__(self, update: Callable[[TensorDictBase], TensorDictBase], batch: TensorDictBase):
        self.update = update
        self.batch = batch.clone()
        self.device = batch.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.graph = None
        self.output = None
        self.warmup = 0

    def __call__(self, batch: TensorDictBase) -> TensorDictBase:
        # A final, short minibatch must keep its own size and optimizer step.
        if batch.batch_size != self.batch.batch_size:
            return self.update(batch)
        with torch.cuda.device(self.device):
            self.batch.copy_(batch)
            if self.warmup < 3:
                self.stream.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(self.stream):
                    output = self.update(self.batch)
                torch.cuda.current_stream(self.device).wait_stream(self.stream)
                self.warmup += 1
                return output
            if self.graph is None:
                self.stream.wait_stream(torch.cuda.current_stream(self.device))
                self.graph = torch.cuda.CUDAGraph()
                # Automatic collection may destroy an older experiment's CUDA
                # graphs. Those CUDA calls are forbidden during this capture.
                gc_enabled = gc.isenabled()
                gc.disable()
                try:
                    with torch.cuda.graph(self.graph, stream=self.stream):
                        self.output = self.update(self.batch)
                finally:
                    if gc_enabled:
                        gc.enable()
            # Capture records CUDA work without executing it. This replay is
            # the update for the current minibatch, including the first time.
            self.graph.replay()
            # Logs retained by BenchMARL must not alias the next replay's output.
            return self.output.clone()


def _can_use_cuda_graphs(algorithm, config, model) -> bool:
    """Keep custom models/optimizers and other algorithms on the upstream path."""
    return (
        type(algorithm) in (IppoConfig, MappoConfig)
        and torch.device(config.train_device).type == "cuda"
        and model.layer_class is torch.nn.Linear
        and model.activation_class in (torch.nn.Tanh, torch.nn.ReLU, torch.nn.ELU, torch.nn.GELU, torch.nn.SiLU)
        and model.norm_class in (None, torch.nn.LayerNorm)
        and not config.adam_extra_kwargs.get("differentiable", False)
        and config.adam_extra_kwargs.get("capturable", True)
    )


class _PPOExperiment(Experiment):
    """Reuse upstream PPO with buffered collection and CUDA minibatch execution."""

    def _setup_algorithm(self):
        super()._setup_algorithm()
        from benchmax.integrations.benchmarl.env import _ResponseCheckpoints

        # Our response observer runs only after an update and does not inspect
        # minibatches or alter losses. Preserve the accelerated path for it;
        # other callbacks retain the full upstream samples and hook behavior.
        minibatch_callbacks = any(type(callback) is not _ResponseCheckpoints for callback in self.callbacks)
        self._updates = {}
        self._indexed_updates = {}
        self._indexed_groups = set()
        self._fused_loss_groups = set()
        self._kernels = None
        if not minibatch_callbacks and torch.version.cuda is not None:
            try:
                from benchmax.integrations.benchmarl import ppo_kernels
            except ImportError:
                pass  # Triton is optional outside the Linux CUDA installation.
            else:
                self._kernels = ppo_kernels
        for group, loss in self.losses.items():
            loss._get_entropy = _categorical_entropy
            if not minibatch_callbacks:
                buffer = self.replay_buffers[group]
                if (
                    type(buffer) is TensorDictReplayBuffer
                    and type(buffer.storage) is LazyTensorStorage
                    and type(buffer.sampler) is SamplerWithoutReplacement
                    and not len(buffer._transform)
                ):
                    self._indexed_groups.add(group)
                # GAE has already computed the targets before buffer.extend().
                # Avoid indexing/copying unused next observations and logging
                # fields for every minibatch. Keep full samples for callbacks.
                keys = list(dict.fromkeys(
                    loss.actor_network.in_keys + loss.critic_network.in_keys
                    + [(group, key) for key in ("action", "log_prob", "advantage", "value_target", "state_value")]
                ))
                self.replay_buffers[group].append_transform(
                    SelectTransform(*keys, keep_rewards=False, keep_dones=False), invert=True
                )
            if (
                self._kernels is not None
                and len(self.group_map[group]) == 1
                and loss.functional
                and loss.reduction == "mean"
                and not loss.normalize_advantage
                and loss.entropy_bonus
                and isinstance(loss.entropy_coeff, torch.Tensor)
                and loss.entropy_coeff.numel() == 1
                and loss._has_critic
                and 0 <= self.algorithm.clip_epsilon < 1
            ):
                # TorchRL keeps this scalar on CPU; the kernel needs a device
                # pointer. It remains a registered buffer with the same value.
                loss.entropy_coeff = loss.entropy_coeff.to(self.config.train_device)
                self._fused_loss_groups.add(group)
            for optimizer in self.optimizers[group].values():
                for parameters in optimizer.param_groups:
                    parameters["capturable"] = True

    def _setup_collector(self):
        if self.config.collect_with_grad:
            return super()._setup_collector()
        from benchmax.integrations.benchmarl.collector import _PPOCollector

        self.policy = self.algorithm.get_policy_for_collection()
        self.group_policies = {
            group: self.policy.select_subsequence(out_keys=[(group, "action")])[0]
            for group in self.group_map
        }
        self.collector = _PPOCollector(
            self.env_func,
            self.policy,
            device=self.config.sampling_device,
            storing_device=self.config.sampling_device,
            frames_per_batch=self.config.collected_frames_per_batch(self.on_policy),
            total_frames=self.config.get_max_n_frames(self.on_policy),
            init_random_frames=0,
        )

    def _update(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        optimizers = self.optimizers[group]
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        if group in self._fused_loss_groups:
            losses = self._kernels.ppo_loss(self.losses[group], batch)
        else:
            losses = self.losses[group](batch)
        # Explicit device metadata lets TensorDict clone diagnostics with one
        # foreach operation instead of a separate launch for every scalar.
        result = losses.detach().to(self.config.train_device)
        losses = self.algorithm.process_loss_vals(group, losses)
        for name, optimizer in optimizers.items():
            losses[name].backward()
            params = [p for pg in optimizer.param_groups for p in pg["params"]]
            if self.config.clip_grad_norm and self.config.clip_grad_val is not None:
                clip_norm = torch.nn.utils.clip_grad_norm_ if self._kernels is None else self._kernels.clip_grad_norm
                norm = clip_norm(params, self.config.clip_grad_val)
            else:
                norm = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in params if p.grad is not None]))
                if self.config.clip_grad_val is not None:
                    torch.nn.utils.clip_grad_value_(params, self.config.clip_grad_val)
            result.set(f"grad_norm_{name}", norm.detach())
            if self._kernels is None:
                optimizer.step()
            else:
                self._kernels.adam_step(optimizer)
        return result

    def _optimizer_loop(self, group: str) -> TensorDictBase:
        buffer = self.replay_buffers[group]
        if group in self._indexed_groups and len(buffer):
            # Use the original sampler and RNG sequence. Storage is populated
            # in place each rollout, so only indices need copying into a graph.
            # This path has no sampling transforms, minibatch callbacks or priorities.
            index, _ = buffer.sampler.sample(buffer.storage, buffer.batch_size)
            indices = TensorDict({"index": index}, batch_size=index.shape, device=index.device)
            if group not in self._indexed_updates:
                source = buffer.storage[:].exclude("index")
                gather = (lambda data, index: data[index]) if self._kernels is None else self._kernels.gather_replay
                self._indexed_updates[group] = _CudaGraphUpdate(
                    lambda data: self._update(group, gather(source, data["index"])), indices
                )
            return self._indexed_updates[group](indices)
        batch = self.replay_buffers[group].sample().to(self.config.train_device)
        if group not in self._updates:
            self._updates[group] = _CudaGraphUpdate(lambda data: self._update(group, data), batch)
        result = self._updates[group](batch)
        self.replay_buffers[group].update_tensordict_priority(batch)
        callback_loss = self._on_train_step(batch, group)
        if callback_loss is not None:
            result.update(callback_loss)
        return result
