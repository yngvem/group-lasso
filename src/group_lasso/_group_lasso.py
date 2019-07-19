from abc import ABC, abstractmethod
from math import sqrt
from numbers import Number
import warnings

import numpy.linalg as la
import numpy as np
from sklearn.utils import (
    check_random_state,
    check_array,
    check_consistent_length,
)
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    RegressorMixin,
    ClassifierMixin,
)

from group_lasso._singular_values import find_largest_singular_value
from group_lasso._subsampling import subsample
from group_lasso._fista import fista


_DEBUG = False


def _l2_prox(w, reg):
    """The proximal operator for reg*||w||_2 (not squared).
    """
    return max(0, 1 - reg / la.norm(w)) * w


def _group_l2_prox(w, reg_coeffs, groups):
    """The proximal map for the specified groups of coefficients.
    """
    w = w.copy()

    for group, reg in zip(groups, reg_coeffs):
        reg = reg * sqrt(group.sum())
        w[group] = _l2_prox(w[group], reg)

    return w


def _split_intercept(w):
    return w[0], w[1:]


def _join_intercept(b, w):
    m, n = w.shape
    return np.concatenate([np.array(b).reshape(1, n), w], axis=0)


def _add_intercept_col(X):
    ones = np.ones([X.shape[0], 1])
    return np.concatenate([ones, X], axis=1)


class BaseGroupLasso(ABC, BaseEstimator, TransformerMixin):
    """
    This class implements the Group Lasso [1]_ regularisation for optimisation
    problems with Lipschitz continuous gradients, which is approximately
    equivalent to having a bounded second derivative.

    The loss is optimised using the FISTA algorithm proposed in [2]_ with the
    generalised gradient-based restarting scheme proposed in [3]_.

    References
    ----------
    .. [1] Yuan M, Lin Y. Model selection and estimation in regression with
       grouped variables. Journal of the Royal Statistical Society: Series B
       (Statistical Methodology). 2006 Feb;68(1):49-67.
    .. [2] Beck A, Teboulle M. A fast iterative shrinkage-thresholding algorithm
       for linear inverse problems. SIAM journal on imaging sciences.
       2009 Mar 4;2(1):183-202.
    .. [3] O’Donoghue B, Candes E. Adaptive restart for accelerated gradient
       schemes. Foundations of computational mathematics.
       2015 Jun 1;15(3):715-32.
    """

    # TODO: Document code

    LOG_LOSSES = False

    def __init__(
        self,
        groups,
        reg=0.05,
        n_iter=100,
        tol=1e-5,
        subsampling_scheme=None,
        fit_intercept=True,
        random_state=None,
    ):
        """

        Arguments
        ---------
        groups : Iterable
            Iterable that specifies which group each column corresponds to.
            For columns that should not be regularised, the corresponding
            group index should either be None or negative. For example, the
            list ``[1, 1, 1, 2, 2, -1]`` specifies that the first three
            columns of the data matrix belong to the first group, the next
            two columns belong to the second group and the last column should
            not be regularised.
        reg : float or iterable [default=0.05]
            The regularisation coefficient(s). If ``reg`` is an
            iterable, then it should have the same length as
            ``groups``.
        n_iter : int [default=100]
            The maximum number of iterations to perform
        tol : float [default=1e-5]
            The convergence tolerance. The optimisation algorithm
            will stop once ||x_{n+1} - x_n|| < ``tol``.
        subsampling_scheme : None, float, int or str [default=None]
            The subsampling rate used for the gradient and singular value
            computations. If it is a float, then it specifies the fraction
            of rows to use in the computations. If it is an int, it
            specifies the number of rows to use in the computation and if
            it is a string, then it must be 'sqrt' and the number of rows used
            in the computations is the square root of the number of rows
            in X.
        frobenius_lipschitz : bool [default=False]
            Use the Frobenius norm to estimate the lipschitz coefficient of the
            MSE loss. This works well for systems whose power iterations
            converge slowly. If False, then subsampled power iterations are
            used. Using the Frobenius approximation for the Lipschitz
            coefficient might fail, and end up with all-zero weights.
        fit_intercept : bool [default=True]
            Whether to fit an intercept or not.
        random_state : np.random.RandomState [default=None]
            The random state used for initialisation of parameters.
        """
        self.groups = groups
        self.reg = reg
        self.n_iter = n_iter
        self.tol = tol
        self.subsampling_scheme = subsampling_scheme
        self.fit_intercept = fit_intercept
        self.random_state = random_state

    def _regularizer(self, w):
        """The regularisation penalty for a given coefficient vector, ``w``.
        """
        regularizer = 0
        b, w = _split_intercept(w)
        for group, reg in zip(self.groups_, self.reg_vector):
            regularizer += reg * la.norm(w[group, :])
        return regularizer

    def _get_reg_vector(self, reg):
        """Get the group-wise regularisation coefficients from ``reg``.
        """
        if isinstance(reg, Number):
            reg = [reg * sqrt(group.sum()) for group in self.groups_]
        else:
            reg = list(reg)
        return reg

    @abstractmethod
    def _unregularised_loss(self, X, y, w):
        """The unregularised reconstruction loss.
        """
        pass

    def _loss(self, X, y, w):
        """The group-lasso regularised loss.

        Arguments
        ---------
        X : np.ndarray
            Data matrix, ``X.shape == (num_datapoints, num_features)``
        y : np.ndarray
            Target vector/matrix, ``y.shape == (num_datapoints, num_targets)``,
            or ``y.shape == (num_datapoints,)``
        w : np.ndarray
            Coefficient vector, ``w.shape == (num_features, num_targets)``,
            or ``w.shape == (num_features,)``
        """
        return self._unregularised_loss(X, y, w) + self._regularizer(w)

    def loss(self, X, y):
        """The group-lasso regularised loss with the current coefficients

        Arguments
        ---------
        X : np.ndarray
            Data matrix, ``X.shape == (num_datapoints, num_features)``
        y : np.ndarray
            Target vector/matrix, ``y.shape == (num_datapoints, num_targets)``,
            or ``y.shape == (num_datapoints,)``
        """
        return self._loss(X, y, self.coef_)

    @abstractmethod
    def _compute_lipschitz(self, X, y):
        """Compute Lipschitz bound for the gradient of the unregularised loss.

        The Lipschitz bound is with respect to the coefficient vector or 
        matrix.
        """
        pass

    @abstractmethod
    def _grad(self, X, y, w):
        """Compute the gradient of the unregularised loss wrt the coefficients.
        """
        pass

    def _minimise_loss(self, X, y, lipschitz=None):
        """Use the FISTA algorithm to solve the group lasso regularised loss.
        """
        if self.fit_intercept:
            X = _add_intercept_col(X)

        if lipschitz is None:
            lipschitz = self._compute_lipschitz(X, y)

        if not self.fit_intercept:
            X = _add_intercept_col(X)

        def grad(w):
            g = self._grad(X, y, w)
            if not self.fit_intercept:
                g[0] = 0
            return g

        def prox(w):
            b, w_ = _split_intercept(w)
            w_ = _group_l2_prox(w_, self.reg_vector, self.groups_)
            return _join_intercept(b, w_)

        def loss(w):
            X_, y_ = self.subsample(X, y)
            self._loss(X_, y_, w)

        def callback(x, it_num, previous_x=None):
            X_, y_ = self.subsample(X, y)
            w = x
            previous_w = previous_x

            if self.LOG_LOSSES:
                self.losses_.append(self._loss(X_, y_, w))

            if previous_w is None and _DEBUG:
                print(f"Starting FISTA: ")
                print(f"\tInitial loss: {self._loss(X_, y_, w)}")

            elif _DEBUG:
                print(f"Completed iteration {it_num}:")
                print(f"\tLoss: {self._loss(X_, y_, w)}")
                print(f"\tWeight difference: {la.norm(w-previous_w)}")
                print(f"\tWeight norm: {la.norm(w)}")
                print(f"\tGrad: {la.norm(grad(w))}")

        weights = np.concatenate([self.intercept_, self.coef_])
        weights = fista(
            weights,
            grad=grad,
            prox=prox,
            loss=loss,
            lipschitz=lipschitz,
            n_iter=self.n_iter,
            tol=self.tol,
            callback=callback,
        )
        self.intercept_, self.coef_ = _split_intercept(weights)

    def _check_valid_parameters(self):
        """Check that the input parameters are valid.
        """
        assert all(reg >= 0 for reg in self.reg_vector)
        assert len(self.reg_vector) == len(np.unique(self.groups))
        assert self.n_iter > 0
        assert self.tol >= 0

    def _prepare_dataset(self, X, y):
        """Ensure that the inputs are valid and prepare them for fit.
        """
        check_consistent_length(X, y)
        check_array(X)
        check_array(y)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        return X, y

    def _init_fit(self, X, y):
        """Initialise model and check inputs.
        """
        X, y = self._prepare_dataset(X, y)
        groups = np.array([-1 if i is None else i for i in self.groups])
        self.random_state_ = check_random_state(self.random_state)
        self.groups_ = [self.groups == u for u in np.unique(groups) if u >= 0]
        self.reg_vector = self._get_reg_vector(self.reg)
        self.losses_ = []
        self.coef_ = self.random_state_.standard_normal(
            (X.shape[1], y.shape[1])
        )
        self.coef_ /= la.norm(self.coef_)
        self.intercept_ = np.zeros((1, self.coef_.shape[1]))

        self._check_valid_parameters()

    def fit(self, X, y, lipschitz=None):
        """Fit a group-lasso regularised linear model.
        """
        self._init_fit(X, y)
        self._minimise_loss(X, y, lipschitz=lipschitz)

    @abstractmethod
    def predict(self, X):
        """Predict using the linear model.
        """
        pass

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)

    @property
    def sparsity_mask(self):
        """A boolean mask indicating whether features are used in prediction.
        """
        pattern = np.zeros(len(self.coef_), dtype=bool)
        for group in self.groups_:
            pattern[group] = not np.allclose(
                self.coef_[group], 0, atol=0, rtol=1e-10
            )

        return pattern

    def transform(self, X):
        """Remove columns corresponding to zero-valued coefficients.
        """
        return X[:, self.sparsity_mask]

    def fit_transform(self, X, y, lipschitz=None):
        """Fit a group lasso model to X and y and remove unused columns from X
        """
        self.fit(X, y, lipschitz)
        return self.transform(X)

    def subsample(self, *args):
        """Subsample the input using this class's subsampling scheme.
        """
        return subsample(
            self.subsampling_scheme, random_state=self.random_state_, *args
        )


def _l2_grad(A, b, x):
    """The gradient of the problem ||Ax - b||^2 wrt x.
    """
    return A.T @ (A @ x - b)


class GroupLasso(BaseGroupLasso, RegressorMixin):
    """
    This class implements the Group Lasso [1]_ regularisation for linear
    regression with the mean squared penalty.

    This class is implemented as both a regressor and a transformation.
    If the ``transform`` method is called, then the columns of the input
    that correspond to zero-valued regression coefficients are dropped.

    The loss is optimised using the FISTA algorithm proposed in [2]_ with the
    generalised gradient-based restarting scheme proposed in [3]_. This
    algorithm is not as accurate as a few other optimisation algorithms,
    but it is extremely efficient and does recover the sparsity patterns.
    We therefore reccomend that this class is used as a transformer to select
    the viable features and that the output is fed into another regression
    algorithm, such as RidgeRegression in scikit-learn.

    References
    ----------
    .. [1] Yuan M, Lin Y. Model selection and estimation in regression with
       grouped variables. Journal of the Royal Statistical Society: Series B
       (Statistical Methodology). 2006 Feb;68(1):49-67.

    .. [2] Beck A, Teboulle M. A fast iterative shrinkage-thresholding 
       algorithm for linear inverse problems. SIAM journal on imaging 
       sciences. 2009 Mar 4;2(1):183-202.

    .. [3] O’donoghue B, Candes E. Adaptive restart for accelerated gradient
       schemes. Foundations of computational mathematics.
       2015 Jun 1;15(3):715-32.
    """

    def __init__(
        self,
        groups=None,
        reg=0.05,
        n_iter=100,
        tol=1e-5,
        subsampling_scheme=None,
        fit_intercept=True,
        frobenius_lipschitz=False,
        random_state=None,
    ):
        """

        Arguments
        ---------
        groups : Iterable
            Iterable that specifies which group each column corresponds to.
            For columns that should not be regularised, the corresponding
            group index should either be None or negative. For example, the
            list ``[1, 1, 1, 2, 2, -1]`` specifies that the first three
            columns of the data matrix belong to the first group, the next
            two columns belong to the second group and the last column should
            not be regularised.
        reg : float or iterable [default=0.05]
            The regularisation coefficient(s). If ``reg`` is an
            iterable, then it should have the same length as
            ``groups``.
        n_iter : int [default=100]
            The maximum number of iterations to perform
        tol : float [default=1e-5]
            The convergence tolerance. The optimisation algorithm
            will stop once ||x_{n+1} - x_n|| < ``tol``.
        subsampling_scheme : None, float, int or str [default=None]
            The subsampling rate used for the gradient and singular value
            computations. If it is a float, then it specifies the fraction
            of rows to use in the computations. If it is an int, it
            specifies the number of rows to use in the computation and if
            it is a string, then it must be 'sqrt' and the number of rows used
            in the computations is the square root of the number of rows
            in X.
        frobenius_lipschitz : bool [default=False]
            Use the Frobenius norm to estimate the lipschitz coefficient of the
            MSE loss. This works well for systems whose power iterations
            converge slowly. If False, then subsampled power iterations are
            used. Using the Frobenius approximation for the Lipschitz
            coefficient might fail, and end up with all-zero weights.
        fit_intercept : bool [default=True]
            Whether to fit an intercept or not.
        random_state : np.random.RandomState [default=None]
            The random state used for initialisation of parameters.
        """
        super().__init__(
            groups=groups,
            reg=reg,
            n_iter=n_iter,
            tol=tol,
            subsampling_scheme=subsampling_scheme,
            fit_intercept=fit_intercept,
            random_state=random_state,
        )
        self.frobenius_lipchitz = frobenius_lipschitz

    def fit(self, X, y, lipschitz=None):
        """Fit a group lasso regularised linear regression model.

        Arguments
        ---------
        X : np.ndarray
            Data matrix
        y : np.ndarray
            Target vector or matrix
        lipschitz : float or None [default=None]
            A Lipshitz bound for the mean squared loss with the given
            data and target matrices. If None, this is estimated.
        """
        super().fit(X, y, lipschitz=lipschitz)

    def predict(self, X):
        """Predict using the linear model.
        """
        return self.intercept_ + X @ self.coef_

    def _unregularised_loss(self, X, y, w):
        X_, y_ = self.subsample(X, y)
        MSE = 0.5 * np.sum((X_ @ w - y_) ** 2) / len(X_)
        return MSE

    def _grad(self, X, y, w):
        X_, y_ = self.subsample(X, y)
        SSE_grad = _l2_grad(X_, y_, w)
        return SSE_grad / len(X_)

    def _compute_lipschitz(self, X, y):
        num_rows, num_cols = X.shape
        if self.frobenius_lipchitz:
            return la.norm(X, "fro") ** 2 / (num_rows * num_cols)

        s_max = find_largest_singular_value(
            X,
            subsampling_scheme=self.subsampling_scheme,
            random_state=self.random_state_,
        )
        SSE_lipschitz = 1.5 * s_max ** 2
        return SSE_lipschitz / num_rows


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _logit(X, w):
    return X @ w


def _logistic_proba(X, w):
    return _sigmoid(_logit(X, w))


def _logistic_cross_entropy(X, y, w):
    p = _logistic_proba(X, w)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


class LogisticGroupLasso(BaseGroupLasso, ClassifierMixin):
    """WARNING: Experimental.
    """

    def _compute_proba(self, X, w):
        return _sigmoid(X @ w)

    def _unregularised_loss(self, X, y, w):
        X_, y_ = self.subsample(X, y)
        return _logistic_cross_entropy(X_, y_, w).sum() / len(X)

    def _grad(self, X, y, w):
        X_, y_ = self.subsample(X, y)
        p = _logistic_proba(X_, w)
        return X_.T @ (p - y_) / len(X_)

    def _compute_lipschitz(self, X, y):
        return np.sqrt(12) * np.linalg.norm(X, "fro") / len(X)

    def predict_proba(self, X):
        return _logistic_proba(X, self.coef_)

    def predict(self, X):
        """Predict using the linear model.
        """
        return self.predict_proba(X) >= 0.5

    def fit(self, X, y, lipschitz=None):
        if y.ndim == 2 and y.shape[1] > 1:
            n = y.shape[1]
            warnings.warn(
                f"You have passed {n} targets to a single class classifier. "
                f"This will simply train {n} different models meaning that "
                f"multiple classes can be predicted as true at once."
            )

        super().fit(X, y, lipschitz=lipschitz)
