import numpy as np
import numpy.linalg as la
import pytest

from group_lasso import _singular_values


np.random.seed(0)
TOL = 0.1
SUBSAMPLED_TOL = 0.1
NUM_COLS, NUM_ROWS = 1000, 50
LARGE_COLS, LARGE_ROWS = 10000, 1000
_singular_values._DEBUG = True


@pytest.fixture
def random_svd():
    X = np.random.randn(NUM_COLS, NUM_ROWS)
    U, s, Vh = la.svd(X)
    s = s ** 2
    return U, s, Vh


@pytest.fixture
def large_random_svd():
    X = np.random.randn(LARGE_COLS, LARGE_ROWS)
    U, s, Vh = la.svd(X, full_matrices=False)
    s = s ** 2
    return U, s, Vh


def generate_matrix_from_svd(U, s, Vh):
    return (U[:, : len(s)] * s) @ Vh[: len(s)]


def test_find_largest_singular_value(random_svd):
    s = random_svd[1]
    X = generate_matrix_from_svd(*random_svd)
    smax = _singular_values.find_largest_singular_value(
        X, random_state=np.random
    )

    assert abs(smax - s[0]) / max(smax, s[0]) < TOL
    assert smax < s[0]


def test_power_iteration(random_svd):
    X = generate_matrix_from_svd(*random_svd)
    v0 = np.random.randn(NUM_ROWS)

    v1, s1 = _singular_values._power_iteration(X, v0)
    v1_ = X.T @ (X @ v0)
    s1_ = la.norm(v1_)
    v1_ /= s1_

    assert abs(s1_ - s1) < 1e-10
    assert np.allclose(v1, v1_)


def test_subsampled_find_largest_singular_value(random_svd):
    s = random_svd[1]
    X = generate_matrix_from_svd(*random_svd)
    smax = _singular_values.find_largest_singular_value(
        X, subsampling_scheme="sqrt", random_state=np.random
    )

    assert (s[0] - smax) / max(smax, s[0]) < SUBSAMPLED_TOL
