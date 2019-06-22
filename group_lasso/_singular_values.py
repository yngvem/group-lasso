import warnings

import numpy as np
import numpy.linalg as la
from math import sqrt

from ._subsampling import subsample, subsampling_fraction

_DEBUG = False
LIPSCHITZ_MAXITS = 20
LIPSCHITS_TOL = 5e-3


def _power_iteration(X, v):
    v = X.T@(X@v)
    s = la.norm(v)
    v /= s
    return v, s


def _subsampled_power_iteration(X, v, subsampling_scheme):
    X_ = subsample(subsampling_scheme, X)
    v, s = _power_iteration(X_, v)

    return v, s/subsampling_fraction(len(X), subsampling_scheme)


def _inverse_power_iteration(X, v, smax_sq):
    v = X.T@(X@v) - smax_sq*v
    s = la.norm(v)
    v /= s
    return v, smax_sq - s


def _subsampled_inverse_power_iteration(X, v, smax_sq, subsampling_scheme):
    X_ = subsample(subsampling_scheme, X)
    v, s = _power_iteration(X_, v)
    return v, s/subsampling_fraction(len(X), subsampling_scheme)


def find_largest_singular_value(
    X, subsampling_scheme=1, maxits=LIPSCHITZ_MAXITS, tol=LIPSCHITS_TOL
):
    """Find the largest singular value of X.
    """
    v = np.random.randn(X.shape[1], 1)
    s = la.norm(v)
    v /= s
    for i in range(maxits):
        s_ = s
        v, s = _subsampled_power_iteration(X, v, subsampling_scheme)

        # Absolute value is necessary because of subsampling
        improvement = abs(s - s_)/max(abs(s), abs(s_))
        if improvement < tol and i > 0:
            return np.sqrt(s)

        if _DEBUG:
            print(f'Finished {i}th power iteration:\n'
                  f'\tL={sqrt(s)}\n'
                  f'\tImprovement: {improvement:03g}')

    warnings.warn(
        f'Could not find an estimate for the largest singular value of X'
        f'with the power method. \n'
        f'Ran for {maxits:d} iterations with a tolerance of {tol:02g}'
        f'Subsampling {"is" if subsampling_scheme != 1 else "is not"} used.',
        RuntimeWarning
    )
    return np.sqrt(s)


def find_smallest_singular_value(
    X,
    subsampling_scheme=1,
    smax_sq=None,
    maxits=LIPSCHITZ_MAXITS,
    tol=LIPSCHITS_TOL
):
    """Find the smallest singular value of X.
    """
    if smax_sq is None:
        smax_sq = find_largest_singular_value(X)**2

    v = np.random.randn(X.shape[1], 1)
    s = la.norm(v)
    v /= s
    for i in range(maxits):
        s_ = s
        v, s = _subsampled_inverse_power_iteration(
            X, v, smax_sq, subsampling_scheme
        )

        # Absolute value is necessary because of subsampling
        improvement = abs(s - s_)/max(abs(s), abs(s_))
        if improvement < tol and i > 0:
            return np.sqrt(s)

        if _DEBUG:
            print(f'Finished {i}th power iteration:\n'
                  f'\tL={sqrt(s)}\n'
                  f'\tImprovement: {improvement:03g}')

    warnings.warn(
        f'Could not find an estimate for the largest singular value of X'
        f'with the power method. \n'
        f'Ran for {maxits:d} iterations with a tolerance of {tol:02g}'
        f'Subsampling {"is" if subsampling_scheme != 1 else "is not"} used.',
        RuntimeWarning
    )
    return np.sqrt(s)
