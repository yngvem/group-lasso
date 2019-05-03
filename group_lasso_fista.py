from math import sqrt
from numbers import Number
import warnings

import numpy.linalg as la
import numpy as np

DEBUG = True
LIPSCHITZ_MAXITS = 50
LIPSCHITS_TOL = 1e-2


def _extract_from_singleton_tuple(inputs):
    if len(inputs) == 1:
        return inputs[0]
    return inputs


def _subsample(rate, *Xs):
    """Subsample along first (0-th) axis of the Xs arrays.
    """
    assert len(Xs) > 0
    assert rate <= 1

    if rate == 1:
        return _extract_from_singleton_tuple(Xs)

    num_rows = len(Xs[0])
    num_subsampled_rows = int(num_rows*rate)
    inds = np.random.choice(len(X), num_subsampled_rows, replace=False)
    inds.sort()

    return _extract_from_singleton_tuple(tuple([X[inds, :] for X in Xs]))


def _find_largest_singular_value(
    X, subsampling_rate=1, maxits=LIPSCHITZ_MAXITS, tol=LIPSCHITS_TOL
):
    """Find the largest singular value of X.
    """
    v = np.random.randn(X.shape[1], 1)
    s = la.norm(v)
    v /= s
    for i in range(maxits):
        # I should double check that this actually works
        X_ = _subsample(subsampling_rate, X)

        v = X_.T@(X_@v)
        s_ = la.norm(v)
        v /= s_
        s_ /= subsampling_rate

        # Absolute value is necessary because of subsampling
        improvement = abs(s_ - s)/max(abs(s_), abs(s))
        s = s_
        if improvement < tol and i > 0:
            return s

        if DEBUG:
            print(f'Finished {i}th power iteration:\n'
                  f'\tL={sqrt(s)}\n'
                  f'\tImprovement: {improvement:03g}')

    warnings.warn(
        f'Could not find an estimate for the largest singular value of X'
        f'with the power method. \n'
        f'Ran for {maxits:d} iterations with a tolerance of {tol:02g}'
        f'Subsampling {"is" if subsampling_rate != 1 else "is not"} used.',
        RuntimeWarning
    )
    return s


def _l2_prox(w, reg):
    """The proximal operator for reg*||w||_2 (not squared).
    """
    return max(0, 1 - reg/la.norm(w))*w


def _l2_grad(A, x, b):
    return A.T@(A@x - b)


def _subsampled_l2_grad(A, x, b, rate):
    A, b = _subsample(rate, A, b)
    return _l2_grad(A, x, b)


def _group_l2_prox(w, reg_coeffs, groups):
    """The proximal map for the specified coefficients.
    """
    w = w.copy()
    for (start, end), reg in zip(groups, reg_coeffs):
        reg = reg*sqrt(end - start)
        w[start:end, :] = _l2_prox(w[start:end, :], reg)
    return w


class GroupLassoRegressor:
    """
    This class implements the Group Lasso [1] penalty for linear regression.
    The loss is optimised using the FISTA algorithm proposed in [2] with the
    generalised gradient-based restarting scheme proposed in [3].


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
    # TODO: Estimate smallest singular value and use adaptive FISTA
    # TODO: Follow the sklearn API
    # TODO: Tests

    def __init__(
        self, groups=None, reg=0.05, n_iter=1000, tol=1e-5, subsampling_rate=1
    ):
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
        reg : float or iterable
            The regularisation coefficient(s). If ``reg`` is an
            iterable, then it should have the same length as
            ``groups``.
        n_iter : int
            The maximum number of iterations to perform
        tol : float
            The convergence tolerance. The optimisation algorithm
            will stop once ||x_{n+1} - x_n|| < ``tol``.
        subsampling_rate : float
            The subsampling rate used for the gradient computations.
            Should be in the range (0, 1]. The gradient will be
            computed from a random matrix of size
            ``subsampling_rate * len(X)``.
        """
        self.groups = groups
        self._reset_groups = False
        self.reg = self._get_reg_vector(reg)
        self.n_iter = n_iter
        self.tol = tol
        self.subsampling_rate = subsampling_rate

    def _regularizer(self, w):
        regularizer = 0
        for (start, end), reg in zip(self.groups, self.reg):
            regularizer += reg*la.norm(w[start:end, :])
        return regularizer

    def _get_reg_vector(self, reg):
        if isinstance(reg, Number):
            return [reg*sqrt(end - start) for start, end in self.groups]
        return reg

    def _loss(self, X, y, w):
        MSE = np.sum((X@w - y)**2)/len(X)
        return MSE + self._regularizer(w)

    def loss(self, X, y):
        return self._loss(X, y, self.coef_)

    def _fista_it(self, u, v, t, L, grad, prox):
        u_ = prox(v - grad(v)/L)
        t_ = 0.5 + 0.5*sqrt(1 + 4*t**2)
        du = u_ - u
        v_ = u_ + du*(t-1)/t_

        if (v - u_).T@(u_ - u) > 0:
            if DEBUG:
                print('Restarting')
            u_, v_, t = self._fista_it(
                self.coef_, self.coef_, 1, L, grad, prox
            )

        u = u_
        t = t_
        v = v_

        return u, v, t

    def _fista(self, X, y, lipschitz_coef=None):
        """Use the FISTA algorithm to solve the group lasso regularised loss.
        """
        num_rows, num_cols = X.shape

        if lipschitz_coef is None:
            lipschitz_coef = _find_largest_singular_value(
                X, subsampling_rate=self.subsampling_rate
            )*1.1/num_rows

        def grad(w):
            return _subsampled_l2_grad(X, w, y, self.subsampling_rate)/num_rows

        def prox(w):
            return _group_l2_prox(w, self.reg, self.groups)

        u = self.coef_
        v = self.coef_
        t = 1

        if DEBUG:
            X_, y_ = _subsample(self.subsampling_rate, X, y)
            print(f'Starting FISTA: ')
            print(f'\tInitial loss: {self.loss(X_, y_)}')

        for i in range(self.n_iter):
            u_, v, t = self._fista_it(u, v, t, lipschitz_coef, grad, prox)

            du = u_ - u
            u = u_
            self.coef_ = u

            stopping_criteria = la.norm(du)/(la.norm(u) + 1e-10)

            if DEBUG:
                X_, y_ = _subsample(self.subsampling_rate, X, y)
                print(f'Completed the {i}th iteration:')
                print(f'\tLoss: {self.loss(X_, y_)}')
                print(f'\tStopping criteria: {stopping_criteria:.5g}')
                print(f'\tWeight norm: {la.norm(self.coef_)}')

            if stopping_criteria < self.tol:
                return

        warnings.warn(
            'The FISTA iterations did not converge to a sufficient minimum.\n'
            f'Your subsampling rate is {self.subsampling_rate:g}, if this is '
            'close to zero, then you cannot expect convergence.\n'
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

        assert all(reg > 0 for reg in self.reg)
        assert self.n_iter > 0
        assert self.tol > 0
        assert self.subsampling_rate > 0 and self.subsampling_rate <= 1

        if len(y.shape) != 1:
            assert y.shape[1] == 1
        else:
            y = y.reshape(-1, 1)

        self.coef_ = np.random.randn(X.shape[1], 1)
        self.coef_ /= la.norm(self.coef_)

    def fit(self, X, y, lipschitz_coef=None):
        self._init_fit(X, y)
        self._fista(X, y, lipschitz_coef=lipschitz_coef)

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

    group_sizes = [np.random.randint(3, 10) for i in range(50)]
    groups = get_groups_from_group_sizes(group_sizes)
    num_coeffs = sum(group_sizes)
    num_datapoints = 10000
    noise_level = 0.5

    print('Generating data')
    X = np.random.randn(num_datapoints, num_coeffs)
    print('Generating coefficients')
    w = generate_group_lasso_coefficients(group_sizes, noise_level=0.05)

    print('Generating targets')
    y = X@w
    y += np.random.randn(*y.shape)*noise_level*y

    gl = GroupLassoRegressor(
        groups=groups, n_iter=50, tol=0.1, reg=0.01, subsampling_rate=1
    )
    print('Starting fit')
    gl.fit(X, y)

    plt.plot(w, '.', label='True weights')
    plt.plot(gl.coef_, '.', label='Estimated weights')
    plt.legend()
    plt.show()
