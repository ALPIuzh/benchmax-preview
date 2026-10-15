# Installation

BenchMAX supports **Python 3.11 and 3.12**. Install from the repository with [uv](https://docs.astral.sh/uv/getting-started/installation/) to use its locked dependencies.

## Build requirements

The native evaluator needs a **C++20 compiler, CMake 3.20 or newer, Boost 1.81 or newer, and OpenMP development files**. The build also needs Python development headers.

On macOS, install the command-line developer tools if needed, then the build libraries:

```sh
xcode-select --install
brew install cmake boost libomp
```

On Linux, install these compiler and development packages through your package manager. Check that Boost meets the 1.81 minimum.

## Create the environment

```sh
git clone {{ repository_url }}.git
cd benchmax-preview
uv sync --locked --python 3.12
```

On macOS, replace the last command with:

```sh
CMAKE_PREFIX_PATH="$(brew --prefix libomp)" uv sync --locked --python 3.12
```

The source for the native evaluator is included in `third_party/exp_a_spiel` and builds during installation.

The base installation supports game loading, evaluation, and tabular CFR/CFR+ training. Check that you can import it:

```sh
uv run python -c 'from benchmax import list_available_games; print(list_available_games())'
```

Run the documentation's Python examples with `uv run python`.

(add-notebooks-or-neural-training)=
## Optional dependencies

For IPPO, MAPPO, IQL, approximate response training, or loading the pretrained experts:

```sh
uv sync --locked --extra training
```

For Jupyter notebooks:

```sh
uv sync --locked --group notebook
uv run jupyter lab
```

Use both flags when you need training and notebooks. Keep them on later `uv sync` calls, and retain `CMAKE_PREFIX_PATH` on macOS when rebuilding.

## Next step

[Choose a game and algorithm](games-and-algorithms.md), or go straight to [your first evaluation](quickstart.md).
