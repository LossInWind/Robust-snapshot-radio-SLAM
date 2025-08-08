from __future__ import annotations

import numpy as np


def matrix_inverse(matrix: np.ndarray) -> np.ndarray:
    """Return the inverse of a square matrix using NumPy.

    Parameters
    ----------
    matrix: np.ndarray
        Square matrix to invert.
    """
    return np.linalg.inv(matrix)


def cholesky_factorization(matrix: np.ndarray) -> tuple[np.ndarray, bool]:
    """Compute the Cholesky factorization if possible.

    Returns (L, flag) where L is lower-triangular and flag indicates success.
    """
    try:
        L = np.linalg.cholesky(matrix)
        return L, True
    except np.linalg.LinAlgError:
        # Not positive definite
        n = matrix.shape[0]
        return np.zeros((n, n), dtype=matrix.dtype), False