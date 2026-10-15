"""Custom games and specialized game representations implemented in BenchMAX.

Games from external libraries are integrated through :doc:`providers </reference/providers>`,
declared in ``benchmax/conf/game``, and loaded with :func:`benchmax.load_game`.
"""

from .matrix import MatrixGame, TwoPlayerZeroSumMatrixGame

__all__ = [
    "MatrixGame",
    "TwoPlayerZeroSumMatrixGame",
]
