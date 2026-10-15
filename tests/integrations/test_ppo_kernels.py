"""Compare fused formulas and replay capture with their independent references."""

import pytest

pytest.importorskip("benchmarl")
pytest.importorskip("triton")

import torch
from torchrl.modules.distributions import MaskedCategorical

from benchmax.integrations.benchmarl.ppo_kernels import (
    _Critic,
    _Objective,
    _ReducedObjective,
    adam_step,
    clip_grad_norm,
    gather_replay,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Fused PPO kernels require CUDA")


@pytest.mark.parametrize("actions", [1, 2, 3, 9, 21])
@pytest.mark.parametrize("clip,entropy", [(0.2, 0.01), (0.15, 0.0), (0.3, 0.2)])
def test_objective_values_and_gradients(actions, clip, entropy):
    torch.manual_seed(42)
    logits = (torch.randn(29, 1, actions, device="cuda") * 3).requires_grad_()
    mask = torch.rand_like(logits) > 0.4
    mask[..., 0] = True
    distribution = MaskedCategorical(logits=logits, mask=mask)
    action = distribution.sample()
    log_prob = distribution.log_prob(action)
    old = log_prob.detach() + torch.linspace(-0.5, 0.5, 29, device="cuda").unsqueeze(-1)
    advantage = torch.randn(29, 1, 1, device="cuda")
    # Include zero advantages and a one-legal-action row.
    advantage[0] = 0
    lw = log_prob - old
    epsilon, coefficient = torch.tensor(clip, device="cuda"), torch.tensor(entropy, device="cuda")
    clipped = lw.clamp(torch.log1p(-epsilon), torch.log1p(epsilon))
    objective = (
        -torch.stack([lw.exp() * advantage.squeeze(-1), clipped.exp() * advantage.squeeze(-1)], -1)
        .min(-1)
        .values.mean()
    )
    entropy_loss = -entropy * distribution.entropy().mean()
    gradients = torch.autograd.grad(objective + entropy_loss, logits)[0]
    candidate = logits.detach().clone().requires_grad_()
    obj, ent, h, log_weight, fraction = _Objective.apply(candidate, mask, action, old, advantage, epsilon, coefficient)
    actual_gradients = torch.autograd.grad(obj.mean() + ent.mean(), candidate)[0]
    torch.testing.assert_close(obj.mean(), objective, rtol=3e-4, atol=3e-6)
    torch.testing.assert_close(ent.mean(), entropy_loss, rtol=3e-4, atol=3e-6)
    torch.testing.assert_close(h, distribution.entropy().flatten(), rtol=3e-4, atol=3e-6)
    torch.testing.assert_close(log_weight, lw.flatten(), rtol=3e-4, atol=3e-6)
    torch.testing.assert_close(fraction, (clipped != lw).float().flatten())
    torch.testing.assert_close(actual_gradients, gradients, rtol=3e-4, atol=1e-6)
    reduced_input = logits.detach().clone().requires_grad_()
    reduced = _ReducedObjective.apply(reduced_input, mask, action, old, advantage, epsilon, coefficient)
    expected = (
        objective, entropy_loss, distribution.entropy().mean(), -lw.mean(),
        (clipped != lw).float().mean(),
        (2 * lw.flatten().logsumexp(0) - (2 * lw.flatten()).logsumexp(0)).exp() / lw.numel(),
    )
    for actual, reference in zip(reduced, expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=3e-4, atol=3e-6)
    reduced_gradients = torch.autograd.grad(reduced[0] + reduced[1], reduced_input)[0]
    torch.testing.assert_close(reduced_gradients, gradients, rtol=3e-4, atol=1e-6)


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"amsgrad": True},
        {"weight_decay": 0.1},
        {"maximize": True, "betas": (0.8, 0.97)},
        {"decoupled_weight_decay": True, "weight_decay": 0.2},
        {"foreach": False},
    ],
)
def test_adam_parameters_and_state_after_200_steps(extra):
    torch.manual_seed(42)
    p = torch.nn.Parameter(torch.randn(33, 19, device="cuda"))
    q = torch.nn.Parameter(p.detach().clone())
    reference = torch.optim.Adam([p], lr=0.001, eps=1e-6, capturable=True, **extra)
    candidate = torch.optim.Adam([q], lr=0.001, eps=1e-6, capturable=True, **extra)
    for i in range(200):
        grad = torch.randn_like(p) * (0.01 if i % 2 else 1)
        p.grad, q.grad = grad.clone(), grad.clone()
        reference.step()
        adam_step(candidate)
    torch.testing.assert_close(p, q, rtol=3e-4, atol=3e-6)
    for key in reference.state[p]:
        torch.testing.assert_close(reference.state[p][key], candidate.state[q][key], rtol=3e-4, atol=3e-6)


@pytest.mark.parametrize("dtype,transpose", [(torch.float64, False), (torch.float32, True)])
def test_adam_unsupported_layout_uses_original_step(dtype, transpose, monkeypatch):
    p = torch.nn.Parameter(torch.ones(3, 5, device="cuda", dtype=dtype))
    if transpose:
        p = torch.nn.Parameter(p.T)
    optimizer = torch.optim.Adam([p], capturable=True)
    calls = []
    original = optimizer.step

    def step():
        calls.append(True)
        return original()

    monkeypatch.setattr(optimizer, "step", step)
    for _ in range(3):
        p.grad = torch.ones_like(p)
        adam_step(optimizer)
    assert len(calls) == 3
    assert optimizer.state[p]["step"].item() == 3


@pytest.mark.parametrize("rows", [1, 7, 400, 8192])
@pytest.mark.parametrize("constant", [False, True])
def test_critic_reductions_and_gradients(rows, constant):
    torch.manual_seed(41)
    value = torch.randn(rows, 1, 1, device="cuda", requires_grad=True)
    target = torch.full_like(value, 0.3) if constant else torch.randn_like(value)
    coefficient = torch.tensor(0.7, device="cuda")
    loss = coefficient * (value - target).square().mean()
    residual = (target - value).var(0, correction=0)
    expected_ev = 1 - residual / (target.var(0, correction=0) + torch.finfo(target.dtype).eps)
    expected_grad = torch.autograd.grad(1.3 * loss, value)[0]
    candidate = value.detach().clone().requires_grad_()
    actual_loss, actual_ev = _Critic.apply(candidate, target, coefficient)
    actual_grad = torch.autograd.grad(1.3 * actual_loss, candidate)[0]
    torch.testing.assert_close(actual_loss, loss, rtol=3e-5, atol=1e-6)
    torch.testing.assert_close(actual_ev, expected_ev, rtol=3e-5, atol=1e-6)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=3e-5, atol=1e-6)


@pytest.mark.parametrize("max_norm", [0.0, 0.3, 1e6])
@pytest.mark.parametrize("special", [None, 0.0, float("inf"), float("nan")])
def test_global_gradient_clipping_matches_torch(max_norm, special):
    torch.manual_seed(13)
    left = [torch.nn.Parameter(torch.zeros(n, device="cuda")) for n in (13, 256, 65536)]
    right = [torch.nn.Parameter(p.detach().clone()) for p in left]
    for p, q in zip(left, right, strict=True):
        p.grad = torch.randn_like(p)
        if special == 0:
            p.grad.zero_()
        elif special is not None:
            p.grad[0] = special
        q.grad = p.grad.clone()
    expected = torch.nn.utils.clip_grad_norm_(left, max_norm)
    actual = clip_grad_norm(right, max_norm)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=1e-6, equal_nan=True)
    for p, q in zip(left, right, strict=True):
        torch.testing.assert_close(q.grad, p.grad, rtol=3e-5, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("dtype,transpose", [(torch.float64, False), (torch.float32, True)])
def test_clipping_falls_back_for_unsupported_gradients(dtype, transpose, monkeypatch):
    value = torch.randn(3, 5, device="cuda", dtype=dtype)
    p = torch.nn.Parameter(value.T if transpose else value)
    p.grad = torch.ones_like(p)
    calls = []
    original = torch.nn.utils.clip_grad_norm_

    def clip(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    clip_grad_norm([p], 0.3)
    assert calls == [True]


@pytest.mark.parametrize("layout", ["dense", "transpose", "double"])
def test_replay_gather_preserves_fields_batch_metadata_and_storage(layout):
    from tensordict import TensorDict

    torch.manual_seed(11)
    observations = torch.randn(19, 1, 5, device="cuda")
    if layout == "transpose":
        observations = torch.randn(5, 1, 19, device="cuda").transpose(0, 2)
    elif layout == "double":
        observations = observations.double()
    source = TensorDict({
        "group": TensorDict({
            "observation": observations,
            "mask": torch.rand(19, 1, 3, device="cuda") > 0.5,
            "action": torch.randint(0, 3, (19, 1), device="cuda"),
        }, batch_size=[19, 1]),
        "state": torch.randn(19, 11, device="cuda"),
    }, batch_size=[19], device="cuda")
    index = torch.tensor([18, 0, 11, 0, 2], device="cuda")
    expected, actual = source[index], gather_replay(source, index)
    assert actual.batch_size == expected.batch_size
    assert actual["group"].batch_size == expected["group"].batch_size
    for key in source.keys(True, True):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    snapshot = source.clone()
    actual["group", "observation"].zero_()
    torch.testing.assert_close(source["group", "observation"], snapshot["group", "observation"], rtol=0, atol=0)
