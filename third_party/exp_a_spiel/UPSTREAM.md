# exp-a-spiel provenance

- Repository: https://github.com/gabrfarina/exp-a-spiel
- Commit: `9412882c24e521fad7cbd96fdd68311d9bf6f203`
- License: MIT (`LICENSE`)

## Copied files

- `csrc/base_state.h`
- `csrc/pttt_state.h`
- `csrc/dh_state.h`
- `csrc/utils.h`
- `csrc/log.h`
- `csrc/traverser.h`
- `csrc/traverser.cpp`
- `LICENSE`

## Deliberately omitted files

All other upstream files were omitted. In particular, the CFR and averaging
implementations (`csrc/cfr.cpp`, `csrc/cfr.h`, `csrc/averager.cpp`, and
`csrc/averager.h`), the command-line program (`csrc/eas.cpp`), the upstream
binding (`csrc/bindings.cpp`), and the `src/eas` Python package and generated
stubs are not vendored or built. The upstream CMake files, project metadata,
examples, documentation, lockfile, and tests are also omitted.

## Local changes

- `traverser.h` uses the system-provided Boost.Unordered headers.
- The unused `Traverser::new_averager` declaration, definition, and averager
  includes were removed.
- `EvExpl` and `ev_and_exploitability` were replaced with a metrics-only path
  returning expected value and the two exploitabilities. It does not copy
  gradients or allocate best-response strategies, and calls `Treeplex::br`
  without a destination strategy.
- The complete information-state matrix method was removed; BenchMAX calls the
  existing single-row encoder for bounded batches.
- The `INFO` macro was made a no-op to silence progress output. `CHECK`,
  `FATAL`, assertions, and the traversal algorithm remain unchanged.
- `src/benchmax/integrations/_exp_a_spiel.cpp` is a BenchMAX-authored private
  binding and is not copied from upstream.
