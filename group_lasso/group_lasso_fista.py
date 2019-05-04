from math import sqrt
from numbers import Number
import warnings

import numpy.linalg as la
import numpy as np

from .singular_values import find_largest_singular_value
from ._subsampling import subsample, subsampling_fraction


DEBUG = True


def _l2_prox(w, reg):
    """The proximal operator for reg*||w||_2 (not squared).
    """
    return max(0, 1 - reg/la.norm(w))*w


def _l2_grad(A, x, b):
    return A.T@(A@x - b)


def _subsampled_l2_grad(A, x, b, subsampling_scheme):
    rate = subsampling_fraction(len(A), subsampling_scheme)
    A, b = subsample(subsampling_scheme, A, b)
    return _l2_grad(A, x, b)/rate


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
    # TODO: Estimate smallest singular value and use improved FISTA iterations
    # The imporved fista iterations are outlined in Faster FISTA by Schoenlieb
    # TODO: Follow the sklearn API
    # TODO: Tests

    def __init__(
        self,
        groups=None,
        reg=0.05,
        n_iter=1000,
        tol=1e-5,
        subsampling_scheme=1,
        sqrt_subsampling=False
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
        subsampling_scheme : float
            The subsampling rate used for the gradient computations.
            Should be in the range (0, 1]. The gradient will be
            computed from a random matrix of size
            ``subsampling_scheme * len(X)``.
        """
        self.groups = groups
        self._reset_groups = False
        self.reg = self._get_reg_vector(reg)
        self.n_iter = n_iter
        self.tol = tol
        self.subsampling_scheme = subsampling_scheme

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
            lipschitz_coef = (find_largest_singular_value(
                X, subsampling_scheme=self.subsampling_scheme
            )**2)*1.5/num_rows

        def grad(w):
            SSE_grad = _subsampled_l2_grad(X, w, y, self.subsampling_scheme)
            return SSE_grad/num_rows

        def prox(w):
            return _group_l2_prox(w, self.reg, self.groups)

        u = self.coef_
        v = self.coef_
        t = 1

        if DEBUG:
            X_, y_ = subsample(self.subsampling_scheme, X, y)
            print(f'Starting FISTA: ')
            print(f'\tInitial loss: {self.loss(X_, y_)}')

        for i in range(self.n_iter):
            u_, v, t = self._fista_it(u, v, t, lipschitz_coef, grad, prox)

            du = u_ - u
            u = u_
            self.coef_ = u

            stopping_criteria = la.norm(du)/(la.norm(u) + 1e-10)

            if DEBUG:
                X_, y_ = subsample(self.subsampling_scheme, X, y)
                print(f'Completed the {i}th iteration:')
                print(f'\tLoss: {self.loss(X_, y_)}')
                print(f'\tStopping criteria: {stopping_criteria:.5g}')
                print(f'\tWeight norm: {la.norm(self.coef_)}')
                print(f'\tGrad: {la.norm(grad(self.coef_))}')

            if stopping_criteria < self.tol:
                return

        warnings.warn(
            'The FISTA iterations did not converge to a sufficient minimum.\n'
            f'You used subsampling then this is expected, otherwise,'
            'try to increase the number of iterations '
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

        assert all(reg >= 0 for reg in self.reg)
        assert len(self.reg) == len(self.groups)
        assert self.n_iter > 0
        assert self.tol > 0

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

