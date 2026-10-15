# Jobs

A `Job` manages a training or evaluation operation on an execution [target](../targets/README.md). Creating a job with `submit=True` records the operation, `run(target)` submits it, and `result()` waits for completion and loads the policy or metrics.

```python
from benchmax import load_game
from benchmax.execution import LocalTarget
from benchmax.training import train

game = load_game("kuhn_poker")
target = LocalTarget(runs="./research-runs")
job = train(game, "cfr", iterations=1_000, seed=7, submit=True)
job.run(target)
policy = job.result()
```

Use `job.progress()` to inspect progress, `job.cancel()` to request cancellation, or `await job.result_async()` to wait asynchronously. Reconnect from another session with `Job.open(job.id, target=target)`.

Once submitted, `job.reference()` identifies the output without waiting for it. A [reference](reference.py) can be passed to another operation or saved for later. The job manages execution, while the reference identifies the result.
