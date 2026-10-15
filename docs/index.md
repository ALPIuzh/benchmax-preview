---
hide-toc: true
---

# BenchMAX

```{container} benchmax-subtitle
Exploitability benchmark for multi-agent reinforcement learning.
```

```{container} benchmax-preview
**October 2026 Early Preview — Official Release Planned for January 2027**
```

BenchMAX is a library for benchmarking reinforcement learning algorithms across multi-agent games, with a focus on the exploitability of the policies they learn.

**Core features:**

- A selection of games and training algorithms, with comparable baselines.
- Exact exploitability computation for selected games.
- Approximate exploitability estimation.
- Integrations with [OpenSpiel](https://github.com/google-deepmind/open_spiel) and [BenchMARL](https://github.com/facebookresearch/BenchMARL).
- Pretrained expert policies.

Use BenchMAX to compare new algorithms against baselines and analyse training dynamics!

## Getting started

Begin with [installation](getting-started/installation.md), choose a [game and algorithm](getting-started/games-and-algorithms.md), or load a [pretrained expert](getting-started/expert-policies.md). [Your first evaluation](getting-started/quickstart.md) compares two simple strategies in Rock–Paper–Scissors.

## Example notebooks

Each example notebook works through one experiment, with its code, results, and figures:

- [Train and evaluate RPS policies](notebooks/rock_paper_scissors.ipynb): compare self-play returns with exploitability, then train with regret matching.
- [Train CFR baselines](notebooks/cfr-convergence.ipynb): measure convergence on Kuhn poker.
- [Clone expert policies](notebooks/mail_behavior_cloning.ipynb): learn from demonstrations and compare imitation with exploitability.

## A minimal workflow

Load Kuhn Poker, train a policy profile with counterfactual regret minimization (CFR), and evaluate it:

```python
from benchmax import load_game
from benchmax.training import train

game = load_game("kuhn_poker")
policy = train(game, "cfr", iterations=1_000, seed=7)
metrics = game.evaluate(policy)

print(f"Exploitability: {metrics['exploitability']:.4f}")
```

To evaluate an existing model, pass your own [batched policy callable](getting-started/quickstart.md#the-policy-interface) to `game.evaluate()`. The [training API](reference/training.md) explains the bundled algorithms and how to use a custom trainer.

## Documentation

- [API reference](reference/index.md): concepts and module interfaces.
- [Training](reference/training.md): bundled algorithms and custom trainers.
- [Configuration](reference/config.md): registered names, parameters, and overrides.
- [CLI](getting-started/cli.md): inspect, train, and manage experiments.

```{toctree}
:hidden:
:caption: Start here

Home <self>
getting-started/installation
getting-started/games-and-algorithms
getting-started/expert-policies
getting-started/quickstart
getting-started/cli
```

```{toctree}
:hidden:
:caption: Example notebooks
:glob:

notebooks/rock_paper_scissors
notebooks/cfr-convergence
notebooks/mail_behavior_cloning
notebooks/*
```

```{toctree}
:hidden:
:caption: API reference
:maxdepth: 2
:titlesonly:

Overview <reference/index>
reference/core
reference/games
reference/policy
reference/experts
reference/training
reference/evaluation
reference/execution
reference/storage
reference/references
reference/utilities
```
