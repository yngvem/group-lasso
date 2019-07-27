from math import sqrt

import numpy as np
import numpy.linalg as la

from group_lasso._subsampling import subsample, subsampling_fraction

_DEBUG = False
LIPSCHITZ_MAXITS = 20
LIPSCHITS_TOL = 5e-3


def _power_iteration(X, v):
    v = X.T @ (X @ v)
    s = la.norm(v)
    v /= s
    return v, s


def _subsampled_power_iteration(X, v, subsampling_scheme, random_state):
    X_ = subsample(subsampling_scheme, X, random_state=random_state)
    v, s = _power_iteration(X_, v)

    return (
        v,
        s
        / subsampling_fraction(
            len(X), subsampling_scheme, random_state=random_state
        ),
    )


def find_largest_singular_value(
    X,
    random_state,
    subsampling_scheme=None,
    maxits=LIPSCHITZ_MAXITS,
    tol=LIPSCHITS_TOL,
):
    """Find the largest singular value of X.
    """
    # TODO: This should be some averaging not max-ing.
    v = random_state.randn(X.shape[1], 1)
    s = la.norm(v)
    v /= s
    for i in range(maxits):
        s_ = s
        v_ = v
        v, s = _subsampled_power_iteration(
            X, v, subsampling_scheme, random_state=random_state
        )

        # Absolute value is necessary because of subsampling
        improvement = abs(s - s_) / max(abs(s), abs(s_))
        if improvement < tol and i > 0:
            return np.sqrt(s)

        if s < s_:
            s = s_
            v = v_

        if _DEBUG:
            print(
                (
                    "Finished {i}th power iteration:\n"
                    "\tL={s}\n"
                    "\tImprovement: {improvement:03g}"
                ).format(i=i, s=sqrt(s), improvement=improvement)
            )
    return np.sqrt(s)
