# Expert policies

`load_expert()` loads a pretrained policy profile, so you can evaluate it or collect demonstrations without training it yourself. The current experts require the [training extra](installation.md#optional-dependencies) and are downloaded from the Hugging Face bucket `ALPIuzh/benchmax-preview`.

## Load an expert

Load the policy for a named game:

```python
from benchmax import load_expert, load_game

game = load_game("kuhn_poker")
expert = load_expert(game)
metrics = game.evaluate(expert)
print(metrics["exploitability"])
```

You can also pass the name directly: `load_expert("kuhn_poker")`.

## Available policies

| Game name | Trained with | Recorded exact exploitability |
| --- | --- | ---: |
| `rock_paper_scissors` | IPPO | 0.014219 |
| `kuhn_poker` | IPPO | 0.011123 |
| `phantom_tic_tac_toe` | MAPPO | 0.117047 |
| `abrupt_phantom_tic_tac_toe` | MAPPO | 0.167838 |
| `dark_hex_3x3` | MAPPO | 0.000168 |
| `abrupt_dark_hex_3x3` | MAPPO | 0.175868 |

Gridworld, Liar's Dice, and Connect Four have no registered expert. Calling `load_expert()` for them raises `PolicyNotFoundError`.

## Reuse the policy

For imitation learning, collect the expert's decisions with the same game:

```python
returns, demonstrations = game.rollout(
    expert, episodes=100, seed=7, collect_data=True,
)
player_0_data = demonstrations[0]
```

Each player's data contains observations, actions, and legal-action masks. The [behavioral cloning notebook](../notebooks/mail_behavior_cloning.ipynb) uses these to train an imitation policy.

Downloads are checked against the pinned references and cached. A fully cached expert can be loaded offline. To choose the cache location, pass `cache_dir="./expert-cache"` to `load_expert()`.

An expert must match the game's settings. Changing the evaluator is allowed, but changing game parameters can make the checkpoint incompatible. For other saved policies, see [references and storage](../reference/storage.md).
