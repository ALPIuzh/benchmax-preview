# API reference

A [Game](core.md) defines the players, observations, actions, and environment. A [PolicyProfile](policy.md) supplies a strategy for each player. BenchMAX allows to train and evaluate policies for various games with multiple algorithms. It also provides reference "expert" policies and tools for learning in multi-agent systems.

## Key concepts

| Term | Meaning |
| --- | --- |
| Environment (`GameInstance`) | Mutable state of one episode, created by `game.new_env()`. |
| Exact tree (`GameTreeNode`) | A branching representation for traversing game histories, available for some games. |
| Observation | Information supplied to a player's policy. |
| Information set | Histories a player cannot distinguish, so the strategy must behave consistently across them. |
| Global state | Optional information for centralized training critics, separate from policy observations. |
| Best response | A strategy maximizing one player's return against fixed opponents. |
| NashConv | Sum of the players' gains from switching to best responses. |
| Exploitability | Half NashConv in the two-player zero-sum cases supported by the evaluator. |
| Job / target | A training or evaluation operation, and where it runs. |
| Artifact / reference | Saved results, and a description locating those results rather than loading them. |
| Worker runtime | How an evaluation kernel executes within an operation, distinct from its job target. |

See [Evaluation](evaluation.md) for metric definitions and [Execution](execution.md) for jobs and runtimes.

## Modules

| Module | Purpose |
| --- | --- |
| [Core](core.md) | Game and environment interfaces. |
| [Games](games.md) | Matrix and tabular game representations. |
| [Policy](policy.md) | Policy profiles and callable adapters. |
| [Experts](experts.md) | Pretrained expert policies. |
| [Training](training.md) | Train policy profiles. |
| [Evaluation](evaluation.md) | Evaluators and metrics. |
| [Execution](execution.md) | Execution targets, jobs, and workers. |
| [Storage](storage.md) | Saved policies and artifacts. |
| [References](references.md) | Identify, load, and save results across jobs, stores, and files. |
| [Utilities](utilities.md) | Configuration, registry, providers, seeding, and types. |
