import numpy.linalg as la
from math import sqrt
import warnings


def _fista_momentum(momentum):
    return 0.5 + 0.5 * sqrt(1 + 4 * momentum ** 2)


def _fista_it(x, momentum_x, momentum, lipschitz, grad, prox):
    new_x = prox(momentum_x - grad(momentum_x) / lipschitz)
    new_momentum = _fista_momentum(momentum)

    dx = new_x - x
    new_momentum_x = new_x + dx * (momentum - 1) / momentum

    if (momentum_x.ravel() - new_x.ravel()).T @ (
        new_x.ravel() - x.ravel()
    ) > 0:
        new_x, new_momentum_x, new_momentum = _fista_it(
            x, x, 1, lipschitz, grad, prox
        )

    return new_x, new_momentum_x, new_momentum


def fista(x0, grad, prox, loss, lipschitz, n_iter=10, tol=1e-6, callback=None):
    """Use the FISTA algorithm to solve the given optimisation problem
    """
    if callback is not None:
        callback(x0, 0)

    optimal_x = x0
    momentum_x = x0
    momentum = 1

    for i in range(n_iter):
        previous_x = optimal_x
        optimal_x, momentum_x, momentum = _fista_it(
            optimal_x, momentum_x, momentum, lipschitz, grad, prox
        )

        if callback is not None:
            callback(optimal_x, i, previous_x=previous_x)

        if la.norm(optimal_x - previous_x) / la.norm(optimal_x + 1e-16) < tol:
            return optimal_x

    warnings.warn(
        "The FISTA iterations did not converge to a sufficient minimum.\n"
        "You used subsampling then this is expected, otherwise,"
        "try to increase the number of iterations "
        "or decreasing the tolerance.",
        RuntimeWarning,
    )

    return optimal_x
