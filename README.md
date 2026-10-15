# BenchMAX

**October 2026 Early Preview — Official Release Planned for January 2027**

BenchMAX is a library for benchmarking reinforcement learning algorithms across multi-agent games, with a focus on the exploitability of the policies they learn.

Core features:

- A selection of games and training algorithms, with comparable baselines.
- Exact exploitability computation for selected games.
- Approximate exploitability estimation.
- Integrations with [OpenSpiel](https://github.com/google-deepmind/open_spiel) and [BenchMARL](https://github.com/facebookresearch/BenchMARL).
- Pretrained expert policies.

Use BenchMAX to compare new algorithms against baselines and analyse training dynamics!

## Overview of the Core API

### Training and Evaluation

BenchMAX allows you to easily load a game by name, create an independent environment, train a policy profile, and evaluate it:

```python
from benchmax import load_game

game = load_game("kuhn_poker")
policy_profile = train(game) # train() can be any function that trains a policy profile

results = game.evaluate(policy_profile)
print(results["nash_conv"], results["exploitability"])
```

Learn more in the [training guide](docs/getting-started/games-and-algorithms.md#choose-a-training-algorithm) and [evaluation guide](docs/getting-started/quickstart.md).

### Pretrained Experts

BenchMAX allows you to load pretrained expert policies and collect demonstration data, for example for imitation learning:

```python
from benchmax import load_expert, load_game

game = load_game("kuhn_poker")
expert = load_expert(game)
returns, data = game.rollout(expert, episodes=256, seed=7, collect_data=True)
observations = data[0]["observations"]
actions = data[0]["actions"]
```

Learn more about available experts and setup [here](docs/getting-started/expert-policies.md).

## Supported Games and Algorithms

BenchMAX currently supports the following games and algorithms:

| Game | Exploitability evaluation |
| ---- | ------------------------- |
| Rock-Paper-Scissors | Yes — exact matrix evaluation |
| Kuhn Poker | Yes — exact OpenSpiel evaluation |
| Phantom Tic-Tac-Toe | Yes — exact traversal (exp-a-spiel) |
| Abrupt Phantom Tic-Tac-Toe | Yes — exact traversal (exp-a-spiel) |
| Dark Hex (3x3) | Yes — exact traversal (exp-a-spiel) |
| Abrupt Dark Hex (3x3) | Yes — exact traversal (exp-a-spiel) |
| Connect Four (6x7) | Lower bound — perfect solver (`pons`) |
| Liar's Dice (two dice, five faces) | Yes — exact public-history traversal |

| Algorithm | Description |
| --------- | ----------- |
| CFR | Tabular counterfactual regret minimization |
| CFR+ | CFR with regret matching plus |
| IPPO | Independent PPO with separate critics |
| MAPPO | Multi-agent PPO with a centralized critic |
| IQL | Independent Q-learning |

Note that this selection of games and algorithms will be expanded prior to the official release.

Learn more about supported games and algorithms [here](docs/getting-started/games-and-algorithms.md).

## License

BenchMAX is licensed under the [Apache License 2.0](LICENSE). Bundled third-party code retains its own license notices.
