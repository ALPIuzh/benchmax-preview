# Your first evaluation

Compare two policy profiles in Rock–Paper–Scissors. Both examples run on a CPU with the [base installation](installation.md), and their answers can be checked by hand.

## Start with a uniform strategy

A **policy profile** supplies a strategy for every player. Here, both players choose uniformly among their legal actions:

```python
from benchmax import load_game


def uniform(player, observations, action_space, legal_action_mask):
    return legal_action_mask / legal_action_mask.sum(axis=1, keepdims=True)


game = load_game("rock_paper_scissors")
metrics = game.evaluate(uniform)
print(f"Uniform: {metrics['exploitability']:.1f}")
```

```text
Uniform: 0.0
```

Each player chooses rock, paper, and scissors with probability $1/3$. Against this mixture, changing strategy cannot improve the expected payoff. The profile is a Nash equilibrium, and its exploitability is zero.

## Make the profile exploitable

Now let both players always choose rock. Continue in the same Python session:

```python
def always_rock(player, observations, action_space, legal_action_mask):
    probabilities = legal_action_mask.astype(float) * 0.0
    probabilities[:, action_space.index(0)] = 1.0  # Rock is action 0 in this game.
    return probabilities


metrics = game.evaluate(always_rock)
print(f"Always rock: {metrics['exploitability']:.1f}")
print(f"NashConv: {metrics['nash_conv']:.1f}")
```

```text
Always rock: 1.0
NashConv: 2.0
```

Rock against rock gives both players a payoff of zero. Either player could switch to paper and earn one instead. Each unilateral improvement is therefore **1**, their sum (NashConv) is **2**, and exploitability is **1**.

The evaluator computes these values exactly from the payoff matrix.

## The policy interface

BenchMAX calls your policy with four arguments:

| Argument | Meaning |
| --- | --- |
| `player` | The player whose strategy is being queried. |
| `observations` | A batch of observations for that player. |
| `action_space` | The ordered actions corresponding to the output columns. |
| `legal_action_mask` | A Boolean array of shape `(batch, actions)`, identifying legal actions in each row. |

Return a probability matrix with the same shape as the mask. Each row must sum to one, probabilities must be finite and nonnegative, and illegal actions must have probability zero.

The uniform profile uses only the legal-action mask. A learned policy can also use `observations` and `player`. The always-rock example is specific to Rock–Paper–Scissors, where action `0` is always legal. A policy for other games must respect their legal-action masks.

```{note}
Evaluation queries strategies at arbitrary information states, including states outside previously sampled trajectories. Supply probabilities for every player, rather than a function that only returns one sampled action.
```

## Continue from here

- [Understand the metrics](../reference/evaluation.md), including their assumptions.
- [Train CFR baselines](../notebooks/cfr-convergence.ipynb) and evaluate the resulting profiles.
- [Rock–Paper–Scissors](../notebooks/rock_paper_scissors.ipynb) for self-play, exploitability, and regret matching.
- [Read the API reference](../reference/core.md) for the complete interfaces.
