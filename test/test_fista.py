import pytest

from group_lasso._fista import FISTAProblem


@pytest.fixture
def smooth_problem_1d():
    def f(x):
        return 0.5 * x ** 2

    def df(x):
        return x

    lipschitz = 1

    return f, df, lipschitz


@pytest.fixture
def no_regulariser():
    def g(x):
        return 0

    def prox(x, L):
        return x

    return g, prox


def test_lipschitz_updates_with_small_initial_guess(
    smooth_problem_1d, no_regulariser
):
    f, df, _ = smooth_problem_1d
    g, prox = no_regulariser

    small_L = 0.01
    x0 = 10

    optimiser = FISTAProblem(
        smooth_loss=f,
        proximable_loss=g,
        smooth_grad=df,
        prox=prox,
        init_lipschitz=small_L,
    )

    optimiser.minimise(x0)
    assert optimiser.lipschitz > small_L
