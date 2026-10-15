"""One-line entry points per algorithm.

These add nothing to :func:`benchmax.training.train` beyond naming the
algorithm.
"""

from __future__ import annotations

import functools

from benchmax.training.api import train

train_ippo = functools.partial(train, algorithm="ippo")
train_mappo = functools.partial(train, algorithm="mappo")
train_iql = functools.partial(train, algorithm="iql")
