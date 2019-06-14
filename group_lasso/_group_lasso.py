from math import sqrt
from numbers import Number
import warnings

import numpy.linalg as la
import numpy as np

from ._singular_values import find_largest_singular_value
from ._subsampling import subsample, subsampling_fraction


_DEBUG = False


def _l2_prox(w, reg):
    """The proximal operator for reg*||w||_2 (not squared).
    """
    return max(0, 1 - reg/la.norm(w))*w


def _l2_grad(A, x, b):
    """The gradient of the problem ||Ax - b||^2 wrt x.
    """
    return A.T@(A@x - b)


def _subsampled_l2_grad(A, x, b, subsampling_scheme):
    """An unbiased estimator for the gradient of ||Ax - b||^2 wrt x.
    """
    rate = subsampling_fraction(len(A), subsampling_scheme)
    A, b = subsample(subsampling_scheme, A, b)
    return _l2_grad(A, x, b)/rate


def _group_l2_prox(w, reg_coeffs, groups):
    """The proximal map for the specified groups of coefficients.
    """
    w = w.copy()
    for (start, end), reg in zip(groups, reg_coeffs):
        reg = reg*sqrt(end - start)
        w[start:end, :] = _l2_prox(w[start:end, :], reg)
    return w


class GroupLasso:
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
    # TODO: Use sklean mixins?
    # TODO: Tests

    def __init__(
        self,
        groups=None,
        reg=0.05,
        n_iter=1000,
        tol=1e-5,
        subsampling_scheme=1,
        sqrt_subsampling=False,
        frobenius_lipschitz=False,
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
            [(0, 5), (3, 8)] are not possible, but the groups
            [(0, 5) ,(5, 8)] are possible.
        reg : float or iterable
            The regularisation coefficient(s). If ``reg`` is an
            iterable, then it should have the same length as
            ``groups``.
        n_iter : int
            The maximum number of iterations to perform
        tol : float
            The convergence tolerance. The optimisation algorithm
            will stop once ||x_{n+1} - x_n|| < ``tol``.
        subsampling_scheme : float, int or str
            The subsampling rate used for the gradient and singular value
            computations. If it is a float, then it specifies the fraction
            of rows to use in the computations. If it is an int, it
            specifies the number of rows to use in the computation and if
            it is a string, then it must be 'sqrt' and the number of rows used
            in the computations is the square root of the number of rows
            in X.
        frobenius_lipschitz : Bool
            Use the Frobenius norm to estimate the lipschitz coefficient of the
            MSE loss. If False, then subsampled power iterations are used.
            Using the Frobenius approximation for the Lipschitz coefficient
            might fail, and end up with all-zero weights.
        """
        self.groups = groups
        self.reg = reg
        self.n_iter = n_iter
        self.tol = tol
        self.subsampling_scheme = subsampling_scheme
        self.frobenius_lipchitz = frobenius_lipschitz

    def get_params(self, deep=True):
        return {
            'groups': self.groups,
            'reg': self.reg,
            'n_iter': self.n_iter,
            'tol': self.tol,
            'subsampling_scheme': self.subsampling_scheme
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _regularizer(self, w):
        regularizer = 0
        for (start, end), reg in zip(self.groups, self.reg_):
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

    def _fista_momentum(self, t):
        return 0.5 + 0.5*sqrt(1 + 4*t**2)

    def _fista_it(self, u, v, t, lipschitz, grad, prox):
        u_ = prox(v - grad(v)/lipschitz)
        t_ = self._fista_momentum(t)

        du = u_ - u
        v_ = u_ + du*(t-1)/t_

        if (v - u_).T@(u_ - u) > 0:
            if _DEBUG:
                print('Restarting')
            u_, v_, t = self._fista_it(
                self.coef_, self.coef_, 1, lipschitz, grad, prox
            )

        u = u_
        t = t_
        v = v_

        return u, v, t

    def _compute_lipschitz(self, X):
        num_rows, num_cols = X.shape
        if self.frobenius_lipchitz:
            return la.norm(X, 'fro')**2/(num_rows*num_cols)

        SSE_lipschitz = 1.5*find_largest_singular_value(
            X, subsampling_scheme=self.subsampling_scheme
        )**2
        return SSE_lipschitz/num_rows

    def _has_converged(self, weights, previous_weights):
        weight_difference = la.norm(weights - previous_weights)
        weight_norms = la.norm(weights + 1e-10)
        stopping_criteria = weight_difference/weight_norms
        return stopping_criteria < self.tol

    def _fista(self, X, y, lipschitz_coef=None):
        """Use the FISTA algorithm to solve the group lasso regularised loss.
        """
        num_rows, num_cols = X.shape

        if lipschitz_coef is None:
            lipschitz_coef = self._compute_lipschitz(X)

        def grad(w):
            SSE_grad = _subsampled_l2_grad(X, w, y, self.subsampling_scheme)
            return SSE_grad/num_rows

        def prox(w):
            return _group_l2_prox(w, self.reg_, self.groups)

        u = self.coef_
        v = self.coef_
        t = 1

        if _DEBUG:
            X_, y_ = subsample(self.subsampling_scheme, X, y)
            print(f'Starting FISTA: ')
            print(f'\tInitial loss: {self.loss(X_, y_)}')
            self._losses = []

        for i in range(self.n_iter):
            previous_u = u
            u, v, t = self._fista_it(
                u,
                v,
                t,
                lipschitz_coef,
                grad,
                prox
            )
            self.coef_ = u

            if _DEBUG:
                X_, y_ = subsample(self.subsampling_scheme, X, y)
                print(f'Completed the {i}th iteration:')
                print(f'\tLoss: {self.loss(X_, y_)}')
                print(f'\tWeight difference: {la.norm(u-previous_u)}')
                print(f'\tWeight norm: {la.norm(self.coef_)}')
                print(f'\tGrad: {la.norm(grad(self.coef_))}')
                self._losses.append(self.loss(X_, y_))

            if self._has_converged(u, previous_u):
                return

        warnings.warn(
            'The FISTA iterations did not converge to a sufficient minimum.\n'
            f'You used subsampling then this is expected, otherwise,'
            'try to increase the number of iterations '
            'or decreasing the tolerance.',
            RuntimeWarning
        )

    def _init_fit(self, X, y):
        self.reg_ = self._get_reg_vector(self.reg)

        assert all(reg >= 0 for reg in self.reg_)
        assert len(self.reg_) == len(self.groups)
        assert self.n_iter > 0
        assert self.tol >= 0
        for group1, group2 in zip(self.groups[:-1], self.groups[1:]):
            assert group1[0] < group1[1]
            assert group1[1] <= group2[0]

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
