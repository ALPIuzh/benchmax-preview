# Games and algorithms

Start with Rock–Paper–Scissors or Kuhn Poker to try a complete training and evaluation workflow. Both are small enough to evaluate exactly on a CPU.

## Choose a game

Pass the game's name to `load_game()`:

```python
from benchmax import load_game

game = load_game("kuhn_poker")
```

These named configurations all have two players and zero-sum rewards.

| Name | Game | Default evaluation |
| --- | --- | --- |
| `rock_paper_scissors` | Simultaneous Rock–Paper–Scissors. | Exact payoff matrix. |
| `kuhn_poker` | Poker with private cards and a three-card deck. | Exact game tree. |
| `gridworld_3x3` | Simultaneous pursuit on a 3×3 grid, with discount 0.9. | Exact discounted model. |
| `phantom_tic_tac_toe` | Hidden-board Tic-Tac-Toe where a collision keeps the turn. | Exact native traversal. |
| `abrupt_phantom_tic_tac_toe` | Phantom Tic-Tac-Toe where a collision passes the turn. | Exact native traversal. |
| `dark_hex_3x3` | Hidden-board Hex where a collision keeps the turn. | Exact native traversal. |
| `abrupt_dark_hex_3x3` | Dark Hex where a collision passes the turn. | Exact native traversal. |
| `liars_dice_2d5f` | Liar's Dice with two five-sided dice per player. | Exact native traversal. |
| `connect_four` | Connect Four on a 6×7 board. | Matches against the perfect Pons opponent, giving a sampled exploitability lower bound. |

Exact evaluation of the hidden-board games and Liar's Dice can be expensive. Connect Four's default evaluator needs a separate engine installation:

```sh
uv run benchmax setup pons --rows 6 --columns 7
```

BenchMAX also loads other OpenSpiel games through names such as `openspiel.leduc_poker`. These do not necessarily have a configured evaluator. See [game discovery](../reference/registry.md) for loading options.

## Choose a training algorithm

| Name | Method | Installation | Budget |
| --- | --- | --- | --- |
| `cfr` | Tabular counterfactual regret minimization. | Base | `iterations` |
| `cfr_plus` | CFR with regret matching plus. | Base | `iterations` |
| `ippo` | PPO with independent critics. | Training extra | `total_frames` |
| `mappo` | PPO with a centralized critic. | Training extra | `total_frames` |
| `iql` | Independent Q-learning. | Training extra | `total_frames` |

For a small CPU example:

```python
from benchmax.training import train

policy = train(game, "cfr", iterations=1_000, seed=7)
metrics = game.evaluate(policy)
print(metrics["exploitability"])
```

`train()` returns strategies for all players. You can instead start with a [pretrained expert](expert-policies.md).

## Use a tuned preset

Bare names such as `"ippo"` use BenchMARL's defaults. The `presets` entries provide separate settings for each game:

```python
policy = train(game, "presets.ippo_kuhn_poker", seed=7)
```

This example requires the training extra and defaults to **10 million combined game steps**. The shipped IPPO, MAPPO, and IQL presets cover the first seven games in the table, but Connect Four and Liar's Dice have no tuned preset. You can override the budget or parameters. The [preset YAML files]({{ repository_url }}/tree/main/src/benchmax/conf/algorithm/presets) record the selection and settings.

To list the configured names:

```python
from benchmax import list_available_games, list_available_algorithms

print(list_available_games())
print(list_available_algorithms())
```

Listing algorithms does not require their optional dependencies. For training options and saving policies, see [train a baseline](../reference/training.md).
