"""Fused float32 CUDA kernels for the adapter's discrete PPO path.

Imported lazily: CPU training and installations without Triton keep TorchRL and
PyTorch implementations. These kernels change operation grouping, not PPO or
Adam hyperparameters. Unsupported reductions and layouts stay in PyTorch.
"""

import math

import torch
import triton
import triton.language as tl
from tensordict import TensorDict
from triton.language.extra.cuda import libdevice


@triton.jit
def _objective_forward(L, M, A, OLD, ADV, EPS, COEF, OBJ, ENT, H, LW, CF, GO, GE, C: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < C
    mask = tl.load(M + row * C + cols, valid, other=0)
    logits = tl.load(L + row * C + cols, valid, other=0)
    logits = tl.where(mask, logits, float("-inf"))
    maximum = tl.max(logits, 0)
    exps = tl.exp(logits - maximum)
    denom = tl.sum(exps, 0)
    probs = exps / denom
    lp = logits - maximum - tl.log(denom)
    entropy = -tl.sum(tl.where(mask, probs * lp, 0), 0)
    action = tl.load(A + row)
    chosen = tl.sum(tl.where(cols == action, lp, 0), 0)
    lw = chosen - tl.load(OLD + row)
    eps = tl.load(EPS)
    clipped = tl.minimum(tl.maximum(lw, tl.log(1 - eps)), tl.log(1 + eps))
    adv = tl.load(ADV + row)
    gain1 = tl.exp(lw) * adv
    gain2 = tl.exp(clipped) * adv
    # torch.stack(...).min selects the first entry at ties. Within the
    # clipping interval both branches have the same derivative.
    grad_obj = tl.where(gain1 <= gain2, -gain1, 0) * ((cols == action).to(tl.float32) - probs)
    coef = tl.load(COEF)
    grad_entropy = tl.where(mask, coef * probs * (lp + entropy), 0)
    tl.store(OBJ + row, -tl.minimum(gain1, gain2))
    tl.store(ENT + row, -coef * entropy)
    tl.store(H + row, entropy)
    tl.store(LW + row, lw)
    tl.store(CF + row, (clipped != lw).to(tl.float32))
    tl.store(GO + row * C + cols, grad_obj, valid)
    tl.store(GE + row * C + cols, grad_entropy, valid)


@triton.jit
def _objective_backward(
    GO, GE, DO, DE, OUT, N, C: tl.constexpr, SO: tl.constexpr, SE: tl.constexpr, BLOCK: tl.constexpr
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = idx < N
    row = idx // C
    a = tl.load(GO + idx, valid, other=0) * tl.load(DO + row * SO, valid, other=0)
    b = tl.load(GE + idx, valid, other=0) * tl.load(DE + row * SE, valid, other=0)
    tl.store(OUT + idx, a + b, valid)


class _Objective(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, mask, actions, old, advantage, epsilon, coefficient):
        n, c = logits.numel() // logits.shape[-1], logits.shape[-1]
        out = [torch.empty(n, device=logits.device, dtype=logits.dtype) for _ in range(5)]
        go, ge = torch.empty_like(logits), torch.empty_like(logits)
        _objective_forward[(n,)](
            logits,
            mask,
            actions,
            old,
            advantage,
            epsilon,
            coefficient,
            *out,
            go,
            ge,
            C=c,
            BLOCK=triton.next_power_of_2(c),
            num_warps=1,
            enable_fp_fusion=False,
        )
        ctx.save_for_backward(go, ge)
        ctx.mark_non_differentiable(*out[2:])
        return tuple(out)

    @staticmethod
    def backward(ctx, do, de, *unused):
        go, ge = ctx.saved_tensors
        # Mean reductions produce zero-stride gradients: read their scalar
        # directly instead of materializing two expanded vectors.
        out = torch.empty_like(go)
        _objective_backward[(triton.cdiv(go.numel(), 256),)](
            go,
            ge,
            do,
            de,
            out,
            go.numel(),
            go.shape[-1],
            SO=do.stride(0),
            SE=de.stride(0),
            BLOCK=256,
            enable_fp_fusion=False,
        )
        return out, None, None, None, None, None, None


@triton.jit
def _objective_statistics(O, E, H, LW, CF, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    mask = i < N
    o = tl.load(O + i, mask, other=0)
    e = tl.load(E + i, mask, other=0)
    h = tl.load(H + i, mask, other=0)
    lw = tl.load(LW + i, mask, other=float("-inf"))
    cf = tl.load(CF + i, mask, other=0)
    maximum = tl.max(lw, 0)
    lse = maximum + tl.log(tl.sum(tl.exp(lw - maximum), 0))
    lse2 = 2 * maximum + tl.log(tl.sum(tl.exp(2 * (lw - maximum)), 0))
    tl.store(OUT, tl.sum(o, 0) / N)
    tl.store(OUT + 1, tl.sum(e, 0) / N)
    tl.store(OUT + 2, tl.sum(h, 0) / N)
    tl.store(OUT + 3, -tl.sum(tl.where(mask, lw, 0), 0) / N)
    tl.store(OUT + 4, tl.sum(cf, 0) / N)
    tl.store(OUT + 5, tl.exp(2 * lse - lse2) / N)


@triton.jit
def _reduced_objective_backward(GO, GE, DO, DE, OUT, N: tl.constexpr, ROWS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(GO + i, i < N, other=0) * tl.load(DO)
    b = tl.load(GE + i, i < N, other=0) * tl.load(DE)
    tl.store(OUT + i, (a + b) / ROWS, i < N)


class _ReducedObjective(torch.autograd.Function):
    """Combine scalar PPO diagnostics and their mean-gradient scaling."""

    @staticmethod
    def forward(ctx, *args):
        outputs = _Objective.forward(ctx, *args)
        logits = args[0]
        reduced = torch.empty(6, device=logits.device, dtype=logits.dtype)
        n = outputs[0].numel()
        _objective_statistics[(1,)](*outputs, reduced, n, triton.next_power_of_2(n), enable_fp_fusion=False)
        result = tuple(reduced.unbind())
        ctx.mark_non_differentiable(*result[2:])
        return result

    @staticmethod
    def backward(ctx, do, de, *unused):
        go, ge = ctx.saved_tensors
        out = torch.empty_like(go)
        _reduced_objective_backward[(triton.cdiv(go.numel(), 256),)](
            go, ge, do, de, out, go.numel(), go.numel() // go.shape[-1], 256, enable_fp_fusion=False
        )
        return out, None, None, None, None, None, None


def ppo_loss(loss, batch):
    """Reuse functional networks and preserve PPO losses and diagnostics."""
    td = batch.clone(False)
    group = loss.tensor_keys.action[0]
    with loss.actor_network_params.to_module(loss.actor_network):
        loss.actor_network.get_dist_params(td)
    inputs = [
        td[key]
        for key in (
            (group, "logits"),
            (group, "action_mask"),
            loss.tensor_keys.action,
            loss.tensor_keys.sample_log_prob,
            loss.tensor_keys.advantage,
        )
    ]
    logits, mask, actions, old, advantage = inputs
    rows = logits.numel() // logits.shape[-1]
    # This adapter has one agent per group and dense, scalar categorical
    # actions. Preserve the upstream path if its layout ever changes.
    if (
        rows > 8192
        or logits.dtype != torch.float32
        or logits.shape[-2] != 1
        or actions.dtype != torch.int64
        or mask.dtype != torch.bool
        or mask.shape != logits.shape
        or any(x.numel() != rows for x in (actions, old, advantage))
        or logits.shape[-1] > 1024
        or old.requires_grad
        or advantage.requires_grad
        or old.dtype != torch.float32
        or advantage.dtype != torch.float32
        or any(not x.is_contiguous() for x in inputs)
    ):
        return loss(batch)
    obj, ent, h, kl, cf, ess = _ReducedObjective.apply(*inputs, loss.clip_epsilon, loss.entropy_coeff)
    critic, value_clip_fraction, explained_variance = _critic_loss(loss, td)
    result = TensorDict(
        {
            "loss_objective": obj,
            "loss_entropy": ent,
            "entropy": h,
            "clip_fraction": cf,
            "kl_approx": kl,
            "loss_critic": critic.mean(),
            "ESS": ess,
        },
        batch_size=[],
    )
    if value_clip_fraction is not None:
        result["value_clip_fraction"] = value_clip_fraction
    if explained_variance is not None:
        result["explained_variance"] = explained_variance
    loss._clear_weakrefs(
        td,
        result,
        "actor_network_params",
        "critic_network_params",
        "target_actor_network_params",
        "target_critic_network_params",
    )
    return result


@triton.jit
def _adam(
    P,
    G,
    M,
    V,
    S,
    MAX,
    N,
    LR: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    EPS: tl.constexpr,
    WD: tl.constexpr,
    DECOUPLED: tl.constexpr,
    MAXIMIZE: tl.constexpr,
    AMSGRAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N
    p = tl.load(P + idx, mask, other=0)
    g = tl.load(G + idx, mask, other=0)
    m = tl.load(M + idx, mask, other=0)
    v = tl.load(V + idx, mask, other=0)
    step = tl.load(S) + 1
    if MAXIMIZE:
        g = -g
    if WD != 0:
        if DECOUPLED:
            p = p * (1 - LR * WD)
        else:
            g = g + WD * p
    m = m + (1 - B1) * (g - m)
    v = v * B2 + (1 - B2) * g * g
    tl.store(M + idx, m, mask)
    tl.store(V + idx, v, mask)
    if AMSGRAD:
        mx = tl.maximum(tl.load(MAX + idx, mask, other=0), v)
        tl.store(MAX + idx, mx, mask)
        v = mx
    bc1 = 1 - libdevice.pow(tl.full((), B1, tl.float32), step)
    bc2 = 1 - libdevice.pow(tl.full((), B2, tl.float32), step)
    denom = (tl.sqrt(v) / tl.sqrt(bc2) + EPS) / (-LR / bc1)
    tl.store(P + idx, p + m / denom, mask)


@torch.no_grad()
def adam_step(optimizer):
    """Update the original Adam state dict, letting PyTorch initialize new state."""
    if (
        type(optimizer) is not torch.optim.Adam
        or optimizer._optimizer_step_pre_hooks
        or optimizer._optimizer_step_post_hooks
    ):
        return optimizer.step()
    parameters = []
    for group in optimizer.param_groups:
        if (
            group["differentiable"]
            or not group["capturable"]
            or group.get("fused")
            or any(
                not isinstance(x, (int, float))
                for x in (group["lr"], *group["betas"], group["eps"], group["weight_decay"])
            )
        ):
            return optimizer.step()
        for p in group["params"]:
            if p.grad is None:
                continue
            state = optimizer.state.get(p)
            if (
                not state
                or p.dtype != torch.float32
                or not p.is_cuda
                or not p.is_contiguous()
                or p.grad.is_sparse
                or not p.grad.is_contiguous()
                or any(not x.is_cuda or x.dtype != torch.float32 or not x.is_contiguous() for x in state.values())
            ):
                return optimizer.step()
            parameters.append((group, p, state))
    steps = []
    for group, p, state in parameters:
        _adam[(triton.cdiv(p.numel(), 512),)](
            p,
            p.grad,
            state["exp_avg"],
            state["exp_avg_sq"],
            state["step"],
            state.get("max_exp_avg_sq", state["exp_avg_sq"]),
            p.numel(),
            LR=group["lr"],
            B1=group["betas"][0],
            B2=group["betas"][1],
            EPS=group["eps"],
            WD=group["weight_decay"],
            DECOUPLED=group.get("decoupled_weight_decay", False),
            MAXIMIZE=group["maximize"],
            AMSGRAD=group["amsgrad"],
            BLOCK=512,
            enable_fp_fusion=False,
        )
        steps.append(state["step"])
    # A separate launch avoids a race between blocks reading the same counter.
    if steps:
        torch._foreach_add_(steps, 1)


@triton.jit
def _critic_forward(V, T, O, EV, G, N: tl.constexpr, C, BLOCK: tl.constexpr):
    C = tl.load(C)
    i = tl.arange(0, BLOCK)
    t = tl.load(T + i, i < N, other=0)
    v = tl.load(V + i, i < N, other=0)
    r = v - t
    loss = tl.sum(r * r, 0) / N * C
    tm = tl.sum(t, 0) / N
    rm = tl.sum(r, 0) / N
    tv = tl.sum(tl.where(i < N, (t - tm) * (t - tm), 0), 0) / N
    rv = tl.sum(tl.where(i < N, (r - rm) * (r - rm), 0), 0) / N
    tl.store(O, loss)
    tl.store(EV, 1.0 - rv / (tv + 1.1920928955078125e-7))
    tl.store(G + i, 2.0 * C / N * r, i < N)


@triton.jit
def _critic_backward(G, D, O, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(O + i, tl.load(G + i, i < N, other=0) * tl.load(D), i < N)


class _Critic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, t, c):
        out = torch.empty((), device=v.device, dtype=v.dtype)
        ev = torch.empty((1, 1), device=v.device, dtype=v.dtype)
        g = torch.empty_like(v)
        _critic_forward[(1,)](v, t, out, ev, g, v.numel(), c, triton.next_power_of_2(v.numel()), enable_fp_fusion=False)
        ctx.save_for_backward(g)
        ctx.mark_non_differentiable(ev)
        return out, ev

    @staticmethod
    def backward(ctx, do, de):
        (g,) = ctx.saved_tensors
        out = torch.empty_like(g)
        _critic_backward[(triton.cdiv(g.numel(), 256),)](g, do, out, g.numel(), 256, enable_fp_fusion=False)
        return out, None, None


def _critic_loss(self, td):
    # Fuse the adapter's usual L2 loss and population variances. Keep all
    # other critic options (including value clipping) in TorchRL.
    target = td.get(self.tensor_keys.value_target)
    if (
        target is None
        or self.clip_value is not None
        or self.separate_losses
        or self.loss_critic_type != "l2"
        or target.dtype != torch.float32
        or not target.is_cuda
        or self.critic_coeff.requires_grad
        or self.critic_coeff.device != target.device
        or self.critic_coeff.dtype != target.dtype
        or self.critic_coeff.numel() != 1
        or target.requires_grad
        or not target.is_contiguous()
        or target.numel() > 8192
        or target.shape[1:] != torch.Size([1, 1])
    ):
        return self.loss_critic(td)
    with self.critic_network_params.to_module(self.critic_network):
        self.critic_network(td)
    value = td.get(self.tensor_keys.value)
    if value.dtype != target.dtype or value.shape != target.shape or not value.is_contiguous():
        return self.loss_critic(td)
    result, ev = _Critic.apply(value, target, self.critic_coeff)
    self._clear_weakrefs(
        td,
        "actor_network_params",
        "critic_network_params",
        "target_actor_network_params",
        "target_critic_network_params",
    )
    return result, None, ev if self.log_explained_variance else None


# Runtime lengths let games with different observation/action widths reuse
# compiled kernels. Only the structural parameter count remains specialized.
@triton.jit(do_not_specialize=["SIZES"])
def _partial_norm(G, SIZES, PARTS, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = 0
    for i in tl.static_range(len(SIZES)):
        chunks = tl.cdiv(SIZES[i], BLOCK)
        if (pid >= start) & (pid < start + chunks):
            idx = (pid - start) * BLOCK + tl.arange(0, BLOCK)
            x = tl.load(G[i] + idx, idx < SIZES[i], other=0)
            tl.store(PARTS + pid, tl.sum(x * x, 0))
        start += chunks


@triton.jit(do_not_specialize=["SIZES", "NP"])
def _clip_norm(
    G, SIZES, PARTS, NORM, LIMIT: tl.constexpr, NP, PB: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    part_idx = tl.arange(0, PB)
    norm = tl.sqrt(tl.sum(tl.load(PARTS + part_idx, part_idx < NP, other=0), 0))
    coef = tl.minimum(LIMIT / (norm + 1e-6), 1.0, propagate_nan=tl.PropagateNan.ALL)
    if pid == 0:
        tl.store(NORM, norm)
    start = 0
    for i in tl.static_range(len(SIZES)):
        chunks = tl.cdiv(SIZES[i], BLOCK)
        if (pid >= start) & (pid < start + chunks):
            idx = (pid - start) * BLOCK + tl.arange(0, BLOCK)
            x = tl.load(G[i] + idx, idx < SIZES[i], other=0)
            tl.store(G[i] + idx, x * coef, idx < SIZES[i])
        start += chunks


@torch.no_grad()
def clip_grad_norm(parameters, max_norm):
    """Clip the global L2 norm with deterministic block sums, without atomics."""
    parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    if (
        not isinstance(max_norm, (int, float))
        or not math.isfinite(max_norm)
        or not grads
        or len(grads) > 32
        or any(
            g.dtype != torch.float32
            or not g.is_cuda
            or g.is_sparse
            or not g.is_contiguous()
            or g.device != grads[0].device
            or not g.numel()
            for g in grads
        )
    ):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    sizes = tuple(g.numel() for g in grads)
    n = sum(triton.cdiv(n, 1024) for n in sizes)
    if n > 1024:
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    parts = torch.empty(n, device=grads[0].device, dtype=grads[0].dtype)
    norm = torch.empty((), device=grads[0].device, dtype=grads[0].dtype)
    _partial_norm[(n,)](tuple(grads), sizes, parts, 1024, enable_fp_fusion=False)
    _clip_norm[(n,)](
        tuple(grads), sizes, parts, norm, float(max_norm), n, triton.next_power_of_2(n), 1024, enable_fp_fusion=False
    )
    return norm


@triton.jit(do_not_specialize=["WIDTH", "ROWS"])
def _gather(S, D, INDEX, WIDTH, ROWS, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = 0
    for i in tl.static_range(len(WIDTH)):
        count = tl.cdiv(ROWS * WIDTH[i], BLOCK)
        if (pid >= start) & (pid < start + count):
            offset = (pid - start) * BLOCK + tl.arange(0, BLOCK)
            mask = offset < ROWS * WIDTH[i]
            row = tl.load(INDEX + offset // WIDTH[i], mask, other=0)
            value = tl.load(S[i] + row * WIDTH[i] + offset % WIDTH[i], mask, other=0)
            tl.store(D[i] + offset, value, mask)
        start += count


def gather_replay(source, index):
    """Gather valid sampler indices across dense replay fields in one launch."""
    items = list(source.items(True, True))
    if (
        index.dtype != torch.int64
        or not index.is_cuda
        or not index.is_contiguous()
        or index.ndim != 1
        or not index.numel()
        or not items
        or len(items) > 32
        or any(
            not value.is_contiguous()
            or value.device != index.device
            or value.dtype not in (torch.float32, torch.int64, torch.bool)
            for _, value in items
        )
    ):
        return source[index]
    widths = tuple(x[0].numel() for k, x in items)
    if not all(widths):
        return source[index]
    out = source[:1].expand(index.numel()).clone(False)
    tensors = []
    for key, value in items:
        target = torch.empty((index.numel(), *value.shape[1:]), device=value.device, dtype=value.dtype)
        out[key] = target
        tensors.append(target)
    count = sum(triton.cdiv(index.numel() * w, 256) for w in widths)
    _gather[(count,)](
        tuple(value for key, value in items), tuple(tensors), index, widths, index.numel(), 256, enable_fp_fusion=False
    )
    return out
