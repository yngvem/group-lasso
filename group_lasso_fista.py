from math import sqrt
import warnings

import numpy.linalg as la
import numpy as np

DEBUG = True
LIPSCHITZ_MAXITS = 100000
LIPSCHITS_TOL = 1e-2


def _find_largest_singular_value(X):
    v = np.random.randn(X.shape[1], 1)
    s = la.norm(v)
    v /= s
    for i in range(LIPSCHITZ_MAXITS):
        v = X.T@(X@v)
        s_ = la.norm(v)
        v /= s_

        # The absolute value signs should be unnecessary
        # I think the singular value approximation will
        # converge monotonically, but I'll add them for
        # safety.
        improvement = abs(s_ - s)/max(abs(s_), abs(s))
        if improvement < LIPSCHITS_TOL and i > 0:
            return s_
        s = s_

        if DEBUG:
            print(f'Finished {i}th power iteration:\n'
                    f'\tL={sqrt(s)}\n'
                    f'\tImprovement: {improvement:03g}')

    raise RuntimeError(
        f'Could not find an estimate for the largest singular value of X'
        f'with the power method. \n'
        f'Ran for {LIPSCHITS_TOL:d} iterations with a tolerance of'
        f'{LIPSCHITS_TOL:02g}')

def _l2_prox(w, reg):
    """The proximal operator for reg*||w||_2 (not squared).
    """
    return max(0, 1 - reg/la.norm(w))*w


class GroupLassoRegressor:
    """
    This class implements the Group Lasso [1] penalty for linear regression.
    The loss is optimised using the FISTA algorithm proposed in [2] with the
    generalised gradient-based restarting scheeme proposed in [3].


    [1]: Yuan M, Lin Y. Model selection and estimation in regression with
         grouped variables. Journal of the Royal Statistical Society: Series B
         (Statistical Methodology). 2006 Feb;68(1):49-67.
    [2]: Beck A, Teboulle M. A fast iterative shrinkage-thresholding algorithm
         for linear inverse problems. SIAM journal on imaging sciences.
         2009 Mar 4;2(1):183-202.
    [3]: O’donoghue B, Candes E. Adaptive restart for accelerated gradient
         schemes. Foundations of computational mathematics.
         2015 Jun 1;15(3):715-32.
    """
    # TODO: Document code
    # TODO: Change groups from list of sets to start and end indices
    # TODO: Estimate smallest singular value and use adaptive FISTA
    # TODO: Accept separate regularisation coefficients for each group
    # TODO: Follow the sklearn API
    # TODO: Tests

    def __init__(self, groups=None, reg=0.05, n_iter=1000, tol=1e-5):
        """

        Arguments
        ---------
        groups : list of tuples
            List of groups parametrised by indices. The group
            (0, 5) denotes the group of the first five regression
            coefficients. The group (5, 8) denotes the group of
            the next three coefficients, and so forth.

            The groups must be non-overlapping, thus the groups
            [(0, 5), (3, 8)] is not possible, whereas the groups
            [(0, 5) ,(5, 8)] is possible.
        """
        self.groups = groups
        self._reset_groups = False
        self.reg = reg
        self.n_iter = n_iter
        self.tol = tol

    def _SSE(self, w):
        return np.sum((self.X@w - self.y)**2)

    def _MSE(self, w):
        return self._SSE(w)/len(self.X)

    def _regularizer(self, w):
        regularizer = 0
        for start, end in self.groups:
            reg = self.reg*sqrt(end - start)
            regularizer += reg*la.norm(w[start:end, :])
        return regularizer

    def _loss(self, w):
        return self._MSE(w) + self._regularizer(w)

    @property
    def loss(self):
        return self._loss(self.coef_)

    def _grad(self, w):
        return self.X.T@(self.X@w - self.y)/len(self.X)

    def _prox(self, w):
        w = w.copy()
        for start, end in self.groups:
            reg = self.reg*sqrt(end - start)
            w[start:end, :] = _l2_prox(w[start:end, :], reg)
        return w

    def _fista_it(self, x, y, t):
        L = self.lipschitz_coef
        x_ = self._prox(y - self._grad(y)/L)
        t_ = 0.5 + 0.5*sqrt(1 + 4*t**2)
        dx = x_ - x

        y = x_ + dx*(t-1)/t_

        x = x_
        t = t_

        return x, y, t

    def _should_restart_momentum(self, x_, y_, x, y):
        # return self._loss(y_) > self._loss(y)
        return (y - x_).T@(x_ - x) > 0

    def fista(self):
        x = self.coef_
        y = self.coef_
        t = 1

        best_loss = self.loss

        for i in range(self.n_iter):

            x_, y_, t = self._fista_it(x, y, t)
            if self._should_restart_momentum(x_, y_, x, y):
                if DEBUG:
                    print('Restarting')
                x_, y_, t = self._fista_it(self.coef_, self.coef_, 1)

            dx = x_ - x
            y = y_
            x = x_

            stopping_criteria = la.norm(dx)/(la.norm(x) + 1e-10)

            if DEBUG:
                print(f'Completed the {i}th iteration:')
                print(f'\tLoss: {self.loss}')
                print(f'\tStopping criteria: {stopping_criteria}')

            if self._loss(x) < best_loss:
                self.coef_ = x

            if stopping_criteria < self.tol:
                return

        warnings.warn(
            'The FISTA iterations did not converge to a sufficient minimum.\n'
            'Try increasing the number of iterations '
            'or decreasing the tolerance.',
            RuntimeWarning
        )

    def _init_fit(self, X, y):
        if self.groups is None or self._reset_groups:
            self._reset_groups = True
            self.groups = [(i, i+1) for i, _ in range(X.shape[1])]
        
        for group1, group2 in zip(self.groups[:-1], self.groups[1:]):
            assert group1[0] < group1[1]
            assert group1[1] <= group2[0]

        assert self.reg >= 0
        assert self.n_iter > 0
        assert self.tol > 0

        if len(y.shape) != 1:
            assert y.shape[1] == 1
        else:
            y = y.reshape(-1, 1)

        self.X = X
        self.y = y
        self.coef_ = np.random.randn(X.shape[1], 1)
        self.coef_ /= la.norm(self.coef_)

        if DEBUG:
            print('Finding Lipschitz coefficient')
        s1 = _find_largest_singular_value(X)
        self.lipschitz_coef = s1 * 1.3 / len(self.X)

    def fit(self, X, y):
        self._init_fit(X, y)
        self.fista()
    
    def predict(self, X):
        return X@self.coef_
    
    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)


def generate_group_lasso_coefficients(
    group_sizes,
    inclusion_probability=0.5,
    included_std=1,
    noise_level=0,
):
    coefficients = []
    for group_size in group_sizes:
        coefficients_ = np.random.randn(group_size, 1)*included_std
        coefficients_ *= (np.random.uniform(0, 1) < inclusion_probability)
        coefficients_ += np.random.randn(group_size, 1)*noise_level
        coefficients.append(coefficients_)

    return np.concatenate(coefficients, axis=0)


def get_groups_from_group_sizes(group_sizes):
    groups = (0, *np.cumsum(group_sizes))
    return list(zip(groups[:-1], groups[1:]))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    np.random.seed(0)

    group_sizes = [np.random.randint(3, 10) for i in range(100)]
    groups = get_groups_from_group_sizes(group_sizes)
    num_coeffs = sum(group_sizes)
    num_datapoints = 100000
    noise_level = 0.5

    X = np.random.randn(num_datapoints, num_coeffs)
    w = generate_group_lasso_coefficients(group_sizes, noise_level=0.05)

    y = X@w
    y += np.random.randn(*y.shape)*noise_level*y

    gl = GroupLassoRegressor(groups=groups, n_iter=100, reg=0.1)
    gl.fit(X, y)

    plt.plot(w, '.', label='True weights')
    plt.plot(gl.coef_, '.', label='Estimated weights')
    plt.legend()
    plt.show()