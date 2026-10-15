# Configuration

BenchMAX builds games and training algorithms from the configuration in this
directory rather than from Python classes.

## Groups

A **group** is one directory of configurations that build objects of one type.

```
conf/
  game/         builds a Game
  algorithm/    builds a configured training callable
  evaluator/    builds an Evaluator
```

Games and algorithms follow the same structure, so `openspiel.dark_hex` and
`benchmarl.mappo` mean the same kind of thing: that library's own entry, with
its own defaults.

`load_game("kuhn_poker")` and `train(game, "mappo")` both mean: take the entry
of that name from its group, apply the caller's overrides, and instantiate it.

## The three kinds of entry

### Raw provider configurations

A subdirectory named after a provider holds that provider's own games or
algorithms, exactly as the provider defines them.

```
game/openspiel/dark_hex.yaml     -> load_game("openspiel.dark_hex")
algorithm/benchmarl/mappo.yaml   -> train(game, "benchmarl.mappo")
```

**A raw configuration may not set `_params_`.** This is enforced when the
configuration is loaded. The guarantee is that someone
who knows OpenSpiel gets from `openspiel.dark_hex` exactly what
`pyspiel.load_game("dark_hex")` gives them, with no BenchMAX opinion applied.
Raw configurations exist to attach things that do not change the underlying
object, such as an evaluator.

A provider entry needs no file. `load_game("openspiel.leduc_poker")` works
without `game/openspiel/leduc_poker.yaml`, because the provider supplies the defaults
and BenchMAX adds nothing.

### Aliases

`_aliases.yaml` gives bare names to raw provider entries. An alias changes
nothing about the object -- same dynamics, same evaluator -- it only makes the
name friendlier, and may freeze parameters at the provider's defaults so the
name cannot come to mean something else.

```yaml
# game/_aliases.yaml
dark_hex_3x3:
  _extends_: openspiel.dark_hex
  _fixed_params_: [board_size, num_rows, num_cols]
```

`load_game("dark_hex_3x3")` is then `load_game("openspiel.dark_hex")`, except
that `params={"board_size": 4}` is refused: a game called `dark_hex_3x3` must
stay 3x3. It cannot be pointed at another game either, because every OpenSpiel
entry inherits `_fixed_: [spiel_name]` from the provider.

### Named configurations

Anything that changes the object gets its own file and its own name. It is not
reachable through a provider namespace, because it is not what the provider
means by that name.

```yaml
# game/abrupt_dark_hex_3x3.yaml
_source_: openspiel.dark_hex
_params_:
  gameversion: adh
_fixed_params_: [board_size, num_rows, num_cols, gameversion]
evaluator:
  _use_: exp_a_spiel
  traverser: AbruptDhEvaluator
```

## Keys

Keys wrapped in underscores belong to the loader, which consumes them. Other
keys are arguments of the thing being built, except `name` inside a group
reference: `evaluator={"name": "approximate_exploitability", "seed": 7}` selects
the registered evaluator and overrides its settings. This is the portable
equivalent of `_use_` for submitted jobs, whose overrides reject loader controls.
Selecting a named reference replaces the previous reference and its settings.
Overrides without a selector continue to merge into the existing settings.

| key | meaning |
| --- | --- |
| `_target_` | what to build, as an import path |
| `_extends_` | start from another entry of this group, **which must exist** |
| `_source_` | start from a provider's own defaults, **whether or not a configuration exists for them** |
| `_use_` | replace this key with the named entry of another group |
| `_params_` | the target's `params` argument: what decides which game or algorithm this is |
| `_fixed_` | arguments of the object that overrides may not change |
| `_fixed_params_` | names under `_params_` that overrides may not change |

Any other key must be an argument of the constructor it builds, so a
configuration that names something the class does not take fails when it is
built.

### `_extends_` versus `_source_`

Both start a configuration from somewhere else, and they differ in what they
require. `_extends_` names an entry that must be loadable, so it fails loudly if
that entry is renamed or removed. `_source_` names a provider entry and falls back
to the provider's own defaults, so it works for the many provider entries that
have no configuration here.

Use `_extends_` when the thing you are extending is a configuration BenchMAX
ships. Use `_source_` when you are deriving from the library.

An entry may use both, and they layer outermost first: the provider's own
defaults, then the BenchMAX configuration being extended, then the entry
itself. That is how a tuned algorithm starts from its library's defaults and
then picks up settings shared across algorithms:

```yaml
# algorithm/mappo.yaml
_source_: benchmarl.mappo    # BenchMARL's own MAPPO
_extends_: _shared           # settings BenchMAX applies to every algorithm
_params_:
  algorithm:                 # and what is specific to this one
    share_param_critic: true
    critic_coef: 1.0
```

### Why `_params_` is separated

Every game and algorithm takes a `params` argument, holding what decides which
game or algorithm it is. `_params_` is that argument: the loader passes the
block through as `params={...}` rather than spreading it over the call. The
other keys are arguments of their own -- `name`, `evaluator`, `description`,
`max_episode_steps` -- and which settings belong on which side is the target's
decision, not the loader's.

Everything under `params` belongs to the game, while everything outside it belongs to
BenchMAX. The two namespaces never merge.

It is also what makes freezing possible. `params` is the only way to a
parameter, so `dark_hex_3x3` cannot be made 4x4 by any spelling:

```python
load_game("dark_hex_3x3", params={"board_size": 4})   # refused: _fixed_params_
load_game("dark_hex_3x3", board_size=4)               # refused: not an argument
```

### Freezing

A name BenchMAX ships means one game, and two keys keep it that way.
`_fixed_params_` freezes names under `_params_`; `_fixed_` freezes the object's
own arguments. Both are needed, because both can change what a name means:

```python
# unfrozen, each of these returns a game still called dark_hex_3x3
load_game("dark_hex_3x3", params={"board_size": 4})   # ...on a 4x4 board
load_game("dark_hex_3x3", spiel_name="kuhn_poker")    # ...that is Kuhn Poker
```

What decides which game or algorithm a provider entry is -- `spiel_name` for
OpenSpiel, `provider` and `name` for BenchMARL -- is frozen once, in the
provider's base configuration, so every entry of that provider inherits it and
no configuration has to remember to.

Both lists **accumulate** as configurations layer. An entry that freezes
something new adds to what it inherits rather than replacing it, so freezing a
parameter cannot quietly unfreeze the game underneath:

```yaml
# game/abrupt_dark_hex_3x3.yaml
_source_: openspiel.dark_hex
_fixed_params_: [board_size, num_rows, num_cols, gameversion]
# resolves to _fixed_params_ as written, and _fixed_: [spiel_name],
# inherited from the provider
```

Only a *caller* is refused. A configuration that derives from a frozen entry
may change what it froze, because what it builds is reached by another name.

## Overrides

Callers pass overrides as keyword arguments, and they merge rather than replace:

```python
load_game("abrupt_dark_hex_3x3", params={"board_size": 4})  # refused: frozen
load_game("openspiel.dark_hex", params={"board_size": 4})   # allowed
load_game("dark_hex_3x3", evaluator="nash_conv")            # allowed
train(game, "mappo", params={"experiment": {"lr": 1e-4}})
```

Because merging recurses, overriding one parameter leaves the others in place.

## Adding a game

- Provided by a library, with its defaults: nothing to do. It already loads as
  `provider.name`. Add a file under `game/<provider>/` only to attach an
  evaluator.
- Provided by a library, and it deserves a shorter name: add an entry to
  `_aliases.yaml`.
- Provided by a library, but with parameters changed: add
  `game/<name>.yaml` with `_source_:` and `_params_:`.
- Implemented in BenchMAX: add `game/<name>.yaml` with `_target_` pointing at
  the class and arguments matching its constructor. Use `_params_` and
  `_fixed_params_` for classes accepting `params`, such as `Gridworld`. Use
  ordinary keys and `_fixed_` for direct arguments, such as `payoffs` on
  `TwoPlayerZeroSumMatrixGame`.
