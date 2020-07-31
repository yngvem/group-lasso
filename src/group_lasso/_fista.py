import warnings
from math import sqrt

import numpy as np
import numpy.linalg as la
from sklearn.exceptions import ConvergenceWarning


class FISTAProblem:
    def __init__(
        self, smooth_loss, proximable_loss, smooth_grad, prox, init_lipschitz
    ):
        self.smooth_loss = smooth_loss
        self.smooth_grad = smooth_grad

        self.proximable_loss = proximable_loss
        self.prox = prox

        self.lipschitz = init_lipschitz

    def _continue_backtracking(self, new_optimal_x, old_momentum_x, lipschitz):
        # Based on FISTA with backtracking. Reformulation of this criterion:
        # F(new_optimal_x) > Q(new_optimal_x, old_momentum_x)
        # f(new_optimal_x) + g(new_optimal_x) > Q(new_optimal_x, old_momentum_x)
        # f(new_optimal_x) > Q(new_optimal_x, old_momentum_x) - g(new_optimal_x)
        # Combine with eq. 2.5 in Beck & Teboulle (2009) to obtain following condition
        # Modified slightly, increasing the threshold for the Lipschitz
        improved_loss = self.smooth_loss(new_optimal_x)
        old_momentum_loss = self.smooth_loss(old_momentum_x)
        old_momentum_grad = self.smooth_grad(old_momentum_x)
        update_vector = new_optimal_x - old_momentum_x

        update_distance = np.sum(update_vector ** 2) * lipschitz / 2.5
        linearised_improvement = (
            old_momentum_grad.ravel().T @ update_vector.ravel()
        )

        return improved_loss > (
            old_momentum_loss + update_distance + linearised_improvement
        )

    def compute_next_momentum(self, current_momentum):
        return 0.5 + 0.5 * sqrt(1 + 4 * current_momentum ** 2)

    def _update_step(self, x, momentum_x, momentum, lipschitz):
        intermediate_step = 0.5 * self.smooth_grad(momentum_x) / lipschitz
        new_x = self.prox(momentum_x - intermediate_step, lipschitz)
        new_momentum = self.compute_next_momentum(momentum)

        dx = new_x - x
        new_momentum_x = new_x + dx * (momentum - 1) / new_momentum

        return new_x, new_momentum_x, new_momentum

    def minimise(self, x0, n_iter=10, tol=1e-6, callback=None):
        """Use the FISTA algorithm to solve the given optimisation problem
        """
        x0 = np.asarray(x0)
        if callback is not None:
            callback(x0, 0)

        optimal_x = x0
        momentum_x = x0
        momentum = 1

        for i in range(n_iter):
            previous_x = optimal_x

            # Simple FISTA update step
            new_optimal_x, new_momentum_x, new_momentum = self._update_step(
                previous_x, momentum_x, momentum, self.lipschitz
            )

            # Adaptive restart criterion from Equation 12 in O’Donoghue & Candès (2012)
            generalised_gradient = momentum_x.ravel() - new_optimal_x.ravel()
            update_vector = new_optimal_x.ravel() - previous_x.ravel()
            # Loss based restart criterion
            if generalised_gradient.T@update_vector > self.smooth_loss(previous_x):
                momentum_x = previous_x
                momentum = 1
                # fmt: off
                new_optimal_x, new_momentum_x, new_momentum = self._update_step(  # noqa: E501
                    previous_x, momentum_x, momentum, self.lipschitz
                )
                # fmt: on

            # Backtracking line search
            while self._continue_backtracking(
                new_optimal_x, momentum_x, self.lipschitz
            ):
                self.lipschitz *= 2
                (
                    new_optimal_x,
                    new_momentum_x,
                    new_momentum,
                ) = self._update_step(
                    optimal_x, momentum_x, momentum, self.lipschitz
                )
            optimal_x, momentum_x, momentum = (
                new_optimal_x,
                new_momentum_x,
                new_momentum,
            )

            if callback is not None:
                callback(optimal_x, i, previous_x=previous_x)

            if (
                la.norm(optimal_x - previous_x) / la.norm(optimal_x + 1e-16)
                < tol
            ):
                return optimal_x

        warnings.warn(
            "The FISTA iterations did not converge to a sufficient minimum.\n"
            "You used subsampling then this is expected, otherwise, "
            "try increasing the number of iterations "
            "or decreasing the tolerance.",
            ConvergenceWarning,
        )

        return optimal_x
