from abc import ABC, abstractmethod
from math import sqrt
from numbers import Number
import warnings

import numpy.linalg as la
import numpy as np
from scipy import sparse
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
from sklearn.preprocessing import LabelBinarizer

from group_lasso._singular_values import find_largest_singular_value
from group_lasso._subsampling import subsample
from group_lasso._fista import fista


_DEBUG = False

_OLD_REG_WARNING = """
The behaviour has changed since v1.1.1, before then, a bug in the optimisation
algorithm made it so the regularisation parameter was scaled by the largest 
eigenvalue of the covariance matrix.

To use the old behaviour, initialise the class with the keyword argument 
`old_regularisation=True`.

To supress this warning, initialise the class with the keyword argument
`supress_warning=True`
"""


def _l1_l2_prox(w, l1_reg, group_reg, groups):
    return _group_l2_prox(_l1_prox(w, l1_reg), group_reg, groups)


def _l1_prox(w, reg):
    return np.sign(w) * np.maximum(0, np.abs(w) - reg)


def _l2_prox(w, reg):
    """The proximal operator for reg*||w||_2 (not squared).
    """
    norm_w = la.norm(w)
    if norm_w == 0:
        return 0 * w
    return max(0, 1 - reg / norm_w) * w


def _group_l2_prox(w, reg_coeffs, groups):
    """The proximal map for the specified groups of coefficients.
    """
    w = w.copy()

    for group, reg in zip(groups, reg_coeffs):
        w[group] = _l2_prox(w[group], reg)

    return w


def _split_intercept(w):
    return w[0], w[1:]


def _join_intercept(b, w):
    m, n = w.shape
    return np.concatenate([np.array(b).reshape(1, n), w], axis=0)


def _add_intercept_col(X):
    ones = np.ones([X.shape[0], 1])
    if sparse.issparse(X):
        return sparse.hstack((ones, X))        
    return np.hstack([ones, X])


class BaseGroupLasso(ABC, BaseEstimator, TransformerMixin):
    """Base class for sparse group lasso regularised optimisation.

    This class implements the Sparse Group Lasso [1] regularisation for
    optimisation problems with Lipschitz continuous gradients, which is
    approximately equivalent to having a bounded second derivative.

    The loss is optimised using the FISTA algorithm proposed in [2] with the
    generalised gradient-based restarting scheme proposed in [3].

    References
    ----------
    [1] Simon, N., Friedman, J., Hastie, T., & Tibshirani, R. (2013).
    A sparse-group lasso. Journal of Computational and Graphical
    Statistics, 22(2), 231-245.

    [2] Beck A, Teboulle M. (2009). A fast iterative shrinkage-thresholding
    algorithm for linear inverse problems. SIAM journal on imaging
    sciences. 2009 Mar 4;2(1):183-202.

    [3] O’Donoghue B, Candes E. (2015) Adaptive restart for accelerated
    gradient schemes. Foundations of computational mathematics.
    Jun 1;15(3):715-32.
    """

    LOG_LOSSES = False

    def __init__(
        self,
        groups,
        group_reg=0.05,
        l1_reg=0.05,
        n_iter=100,
        tol=1e-5,
        subsampling_scheme=None,
        fit_intercept=True,
        random_state=None,
        warm_start=False,
        old_regularisation=False,
        supress_warning=False,
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
        group_reg : float or iterable [default=0.05]
            The regularisation coefficient(s) for the group sparsity penalty.
            If ``group_reg`` is an iterable, then its length should be equal to
            the number of groups.
        l1_reg : float or iterable [default=0.05]
            The regularisation coefficient for the coefficient sparsity
            penalty.
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
        warm_start : bool [default=False]
            If true, then subsequent calls to fit will not re-initialise the
            model parameters. This can speed up the hyperparameter search
        """
        self.groups = groups
        self.group_reg = group_reg
        self.l1_reg = l1_reg
        self.n_iter = n_iter
        self.tol = tol
        self.subsampling_scheme = subsampling_scheme
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.old_regularisation = old_regularisation
        self.warm_start = warm_start
        self.supress_warning = supress_warning

    def _regulariser(self, w):
        """The regularisation penalty for a given coefficient vector, ``w``.

        The first element of the coefficient vector is the intercept which
        is sliced away.
        """
        regulariser = 0
        b, w = _split_intercept(w)
        for group, reg in zip(self.groups_, self.group_reg_vector_):
            regulariser += reg * la.norm(w[group])
        regulariser += self.l1_reg * la.norm(w.ravel(), 1)
        return regulariser

    def _get_reg_vector(self, reg):
        """Get the group-wise regularisation coefficients from ``reg``.
        """
        if isinstance(reg, Number):
            reg = [reg * sqrt(group.sum()) for group in self.groups_]
        else:
            reg = list(reg)
        return reg

    @abstractmethod
    def _unregularised_loss(self, X, y, w):  #  pragma: nocover
        """The unregularised reconstruction loss.
        """
        raise NotImplementedError

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
        return self._unregularised_loss(X, y, w) + self._regulariser(w)

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
    def _compute_lipschitz(self, X, y):  # pragma: nocover
        """Compute Lipschitz bound for the gradient of the unregularised loss.

        The Lipschitz bound is with respect to the coefficient vector or
        matrix.
        """
        raise NotImplementedError

    @abstractmethod
    def _grad(self, X, y, w):  # pragma: nocover
        """Compute the gradient of the unregularised loss wrt the coefficients.
        """
        raise NotImplementedError

    def _unregularised_gradient(self, w):
        g = self._grad(self.X_, self.y_, w)
        if not self.fit_intercept:
            g[0] = 0
        return g

    def _scaled_prox(self, w):
        """Apply the proximal map of the scaled regulariser to ``w``.

        The scaling is the inverse lipschitz coefficient.
        """
        b, w_ = _split_intercept(w)
        l1_reg = self.l1_reg
        group_reg_vector = self.group_reg_vector_
        if not self.old_regularisation:
            l1_reg = l1_reg / self.lipschitz_
            group_reg_vector = group_reg_vector / self.lipschitz_

        w_ = _l1_l2_prox(w_, l1_reg, group_reg_vector, self.groups_)
        return _join_intercept(b, w_)

    def _subsampled_loss(self, w):
        X_, y_ = self.subsample(self.X_, self.y_)
        return self._loss(X_, y_, w)

    def _minimise_loss(self):
        """Use the FISTA algorithm to solve the group lasso regularised loss.
        """
        # Need transition period before the correct regulariser is used without warning
        def callback(x, it_num, previous_x=None):
            X_, y_ = self.subsample(self.X_, self.y_)
            w = x
            previous_w = previous_x

            if self.LOG_LOSSES:
                self.losses_.append(self._loss(X_, y_, w))

            if previous_w is None and _DEBUG:  # pragma: nocover
                print("Starting FISTA: ")
                print(
                    "\tInitial loss: {loss}".format(loss=self._loss(X_, y_, w))
                )

            elif _DEBUG:  # pragma: nocover
                print("Completed iteration {it_num}:".format(it_num=it_num))
                print("\tLoss: {loss}".format(loss=self._loss(X_, y_, w)))
                print(
                    "\tWeight difference: {wdiff}".format(
                        wdiff=la.norm(w - previous_w)
                    )
                )
                print("\tWeight norm: {wnorm}".format(wnorm=la.norm(w)))
                print(
                    "\tGrad: {gnorm}".format(
                        gnorm=la.norm(self._unregularised_gradient(w))
                    )
                )

        weights = np.concatenate([self.intercept_, self.coef_])
        weights = fista(
            weights,
            grad=self._unregularised_gradient,
            prox=self._scaled_prox,
            loss=self._subsampled_loss,
            lipschitz=self.lipschitz_,
            n_iter=self.n_iter,
            tol=self.tol,
            callback=callback,
        )
        self.intercept_, self.coef_ = _split_intercept(weights)

    def _check_valid_parameters(self):
        """Check that the input parameters are valid.
        """
        assert all(reg >= 0 for reg in self.group_reg_vector_)
        groups = np.array(self.groups)
        assert len(self.group_reg_vector_) == len(
            np.unique(groups[groups >= 0])
        )
        assert self.n_iter > 0
        assert self.tol >= 0

    def _prepare_dataset(self, X, y, lipschitz):
        """Ensure that the inputs are valid and prepare them for fit.
        """
        check_consistent_length(X, y)
        X = check_array(X, accept_sparse='csr')
        y = check_array(y)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        # Add the intercept column and compute Lipschitz bound the correct way
        if self.fit_intercept:
            X = _add_intercept_col(X)
            X = check_array(X, accept_sparse='csr')

        if lipschitz is None:
            lipschitz = self._compute_lipschitz(X, y)

        if not self.fit_intercept:
            X = _add_intercept_col(X)
            X = check_array(X, accept_sparse='csr')

        return X, y, lipschitz

    def _init_fit(self, X, y, lipschitz):
        """Initialise model and check inputs.
        """
        self.random_state_ = check_random_state(self.random_state)
        X, y, lipschitz = self._prepare_dataset(X, y, lipschitz)
        groups = np.array([-1 if i is None else i for i in self.groups])

        self.groups_ = [self.groups == u for u in np.unique(groups) if u >= 0]
        self.group_reg_vector_ = self._get_reg_vector(self.group_reg)
        self.losses_ = []

        if not self.warm_start or not hasattr(self, coef_):
            self.coef_ = self.random_state_.standard_normal(
                (X.shape[1] - 1, y.shape[1])
            )
            self.coef_ /= la.norm(self.coef_)
            self.intercept_ = np.zeros((1, self.coef_.shape[1]))

        self._check_valid_parameters()
        self.X_, self.y_, self.lipschitz_ = X, y, lipschitz
        if not self.old_regularisation and not self.supress_warning:
            warnings.warn(_OLD_REG_WARNING)

    def fit(self, X, y, lipschitz=None):
        """Fit a group-lasso regularised linear model.
        """
        self._init_fit(X, y, lipschitz=lipschitz)
        self._minimise_loss()

    @abstractmethod
    def predict(self, X):  # pragma: nocover
        """Predict using the linear model.
        """
        raise NotImplementedError

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)

    @property
    def sparsity_mask(self):
        """A boolean mask indicating whether features are used in prediction.
        """
        coef_ = self.coef_.mean(1)
        mean_abs_coef = abs(coef_.mean())

        return np.abs(coef_) > 1e-10 * mean_abs_coef

    def transform(self, X):
        """Remove columns corresponding to zero-valued coefficients.
        """
        if sparse.issparse(X):
            X = check_array(X, accept_sparse='csc')
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
    """Sparse group lasso regularised least squares linear regression.

    This class implements the Sparse Group Lasso [1] regularisation for
    linear regression with the mean squared penalty.

    This class is implemented as both a regressor and a transformation.
    If the ``transform`` method is called, then the columns of the input
    that correspond to zero-valued regression coefficients are dropped.

    The loss is optimised using the FISTA algorithm proposed in [2] with the
    generalised gradient-based restarting scheme proposed in [3]. This
    algorithm is not as accurate as a few other optimisation algorithms,
    but it is extremely efficient and does recover the sparsity patterns.
    We therefore reccomend that this class is used as a transformer to select
    the viable features and that the output is fed into another regression
    algorithm, such as RidgeRegression in scikit-learn.

    References
    ----------
    [1] Simon, N., Friedman, J., Hastie, T., & Tibshirani, R. (2013).
    A sparse-group lasso. Journal of Computational and Graphical
    Statistics, 22(2), 231-245.

    [2] Beck A, Teboulle M. (2009). A fast iterative shrinkage-thresholding
    algorithm for linear inverse problems. SIAM journal on imaging
    sciences. 2009 Mar 4;2(1):183-202.

    [3] O’Donoghue B, Candes E. (2015) Adaptive restart for accelerated
    gradient schemes. Foundations of computational mathematics.
    Jun 1;15(3):715-32
    """

    def __init__(
        self,
        groups=None,
        group_reg=0.05,
        l1_reg=0.05,
        n_iter=100,
        tol=1e-5,
        subsampling_scheme=None,
        fit_intercept=True,
        frobenius_lipschitz=False,
        random_state=None,
        warm_start=False,
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
        group_reg : float or iterable [default=0.05]
            The regularisation coefficient(s) for the group sparsity penalty.
            If ``group_reg`` is an iterable, then its length should be equal to
            the number of groups.
        l1_reg : float or iterable [default=0.05]
            The regularisation coefficient for the coefficient sparsity
            penalty.
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
        warm_start : bool [default=False]
            If true, then subsequent calls to fit will not re-initialise the
            model parameters. This can speed up the hyperparameter search
        """
        super().__init__(
            groups=groups,
            l1_reg=l1_reg,
            group_reg=group_reg,
            n_iter=n_iter,
            tol=tol,
            subsampling_scheme=subsampling_scheme,
            fit_intercept=fit_intercept,
            random_state=random_state,
            warm_start=warm_start,
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
        MSE = np.sum((X_ @ w - y_) ** 2) / X_.shape[0]
        return 0.5 * MSE

    def _grad(self, X, y, w):
        X_, y_ = self.subsample(X, y)
        SSE_grad = _l2_grad(X_, y_, w)
        return SSE_grad / X_.shape[0]

    def _compute_lipschitz(self, X, y):
        num_rows, num_cols = X.shape
        if self.frobenius_lipchitz:
            if sparse.issparse(X):
                return sparse.linalg.norm(X, "fro")**2/num_rows
            return la.norm(X, "fro") ** 2 / num_rows

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
    """Sparse group lasso regularised single-class logistic regression.

    This class implements the Sparse Group Lasso [1] regularisation for
    logistic regression with a cross entropy penalty.

    This class is implemented as both a regressor and a transformation.
    If the ``transform`` method is called, then the columns of the input
    that correspond to zero-valued regression coefficients are dropped.

    The loss is optimised using the FISTA algorithm proposed in [2] with the
    generalised gradient-based restarting scheme proposed in [3]. This
    algorithm is not as accurate as a few other optimisation algorithms,
    but it is extremely efficient and does recover the sparsity patterns.
    We therefore reccomend that this class is used as a transformer to select
    the viable features and that the output is fed into another classification
    algorithm, such as LogisticRegression in scikit-learn.

    References
    ----------
    [1] Simon, N., Friedman, J., Hastie, T., & Tibshirani, R. (2013).
    A sparse-group lasso. Journal of Computational and Graphical
    Statistics, 22(2), 231-245.

    [2] Beck A, Teboulle M. (2009). A fast iterative shrinkage-thresholding
    algorithm for linear inverse problems. SIAM journal on imaging
    sciences. 2009 Mar 4;2(1):183-202.

    [3] O’Donoghue B, Candes E. (2015) Adaptive restart for accelerated
    gradient schemes. Foundations of computational mathematics.
    Jun 1;15(3):715-32.
    """

    def _compute_proba(self, X, w):
        return _sigmoid(X @ w)

    def _unregularised_loss(self, X, y, w):
        X_, y_ = self.subsample(X, y)
        return _logistic_cross_entropy(X_, y_, w).sum() / X.shape[0]

    def _grad(self, X, y, w):
        X_, y_ = self.subsample(X, y)
        p = _logistic_proba(X_, w)
        return X_.T @ (p - y_) / X_.shape[0]

    def _compute_lipschitz(self, X, y):
        if sparse.issparse(X):
            norm = sparse.linalg.norm(X, "fro")
        else:
            norm = la.norm(X, "fro")
        return np.sqrt(12) * norm / X.shape[0]

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
                (
                    "You have passed {n} targets to a single class classifier."
                    " This will simply train {n} different models meaning that"
                    " multiple classes can be predicted as true at once."
                ).format(n=n)
            )

        super().fit(X, y, lipschitz=lipschitz)


def _softmax(logit):
    logit = logit - logit.max(1, keepdims=True)
    expl = np.exp(logit)
    return expl / expl.sum(axis=(logit.ndim - 1), keepdims=True)


def _softmax_proba(X, W):
    return _softmax(_logit(X, W))


def _softmax_cross_entropy(X, Y, W):
    P = _softmax_proba(X, W)
    return -np.sum(Y * np.log(P))


def _one_hot_encode(y):
    if y.ndim == 1:
        y = LabelBinarizer().fit_transform(y[:, np.newaxis])
    return y


class MultinomialGroupLasso(BaseGroupLasso, ClassifierMixin):
    """Sparse group lasso regularised multi-class logistic regression.

    This class implements the Sparse Group Lasso [1] regularisation for
    multinomial regression (also known as multi-class logistic regression)
    with a cross entropy penalty.

    This class is implemented as both a regressor and a transformation.
    If the ``transform`` method is called, then the columns of the input
    that correspond to zero-valued regression coefficients are dropped.

    The loss is optimised using the FISTA algorithm proposed in [2] with the
    generalised gradient-based restarting scheme proposed in [3]. This
    algorithm is not as accurate as a few other optimisation algorithms,
    but it is extremely efficient and does recover the sparsity patterns.
    We therefore reccomend that this class is used as a transformer to select
    the viable features and that the output is fed into another classification
    algorithm, such as LogisticRegression in scikit-learn.

    References
    ----------
    [1] Simon, N., Friedman, J., Hastie, T., & Tibshirani, R. (2013).
    A sparse-group lasso. Journal of Computational and Graphical
    Statistics, 22(2), 231-245.

    [2] Beck A, Teboulle M. (2009). A fast iterative shrinkage-thresholding
    algorithm for linear inverse problems. SIAM journal on imaging
    sciences. 2009 Mar 4;2(1):183-202.

    [3] O’Donoghue B, Candes E. (2015) Adaptive restart for accelerated
    gradient schemes. Foundations of computational mathematics.
    Jun 1;15(3):715-32
    """

    def __init__(
        self,
        groups,
        group_reg=0.05,
        l1_reg=0.05,
        n_iter=100,
        tol=1e-5,
        subsampling_scheme=None,
        fit_intercept=True,
        random_state=None,
        warm_start=False,
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
        group_reg : float or iterable [default=0.05]
            The regularisation coefficient(s) for the group sparsity penalty.
            If ``group_reg`` is an iterable, then its length should be equal to
            the number of groups.
        l1_reg : float or iterable [default=0.05]
            The regularisation coefficient for the coefficient sparsity
            penalty.
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
        warm_start : bool [default=False]
            If true, then subsequent calls to fit will not re-initialise the
            model parameters. This can speed up the hyperparameter search
        """
        if subsampling_scheme is not None:
            warnings.warn(
                "Subsampling is not stable for multinomial group lasso."
            )
        super().__init__(
            groups=groups,
            group_reg=group_reg,
            l1_reg=l1_reg,
            n_iter=n_iter,
            tol=tol,
            subsampling_scheme=subsampling_scheme,
            fit_intercept=fit_intercept,
            random_state=random_state,
            warm_start=warm_start,
        )

    def _compute_proba(self, X, w):
        return _softmax_proba(X, w)

    def _unregularised_loss(self, X, y, w):
        y = _one_hot_encode(y)
        X_, y_ = self.subsample(X, y)
        return _softmax_cross_entropy(X_, y_, w).sum() / X.shape[0]

    def _grad(self, X, y, w):
        y = _one_hot_encode(y)
        X_, y_ = self.subsample(X, y)
        p = _softmax_proba(X_, w)

        return X_.T @ (p - y_) / X_.shape[0]

    def _compute_lipschitz(self, X, y):
        if sparse.issparse(X):
            norm = sparse.linalg.norm(X, "fro")
        else:
            norm = la.norm(X, "fro")

        C = y.shape[-1]
        return 2 * C ** (1 / 4) * norm / X.shape[0]

    def predict_proba(self, X):
        return _softmax_proba(X, self.coef_)

    def predict(self, X):
        """Predict using the linear model.
        """
        return np.argmax(self.predict_proba(X), axis=1)

    def _prepare_dataset(self, X, y, lipschitz):
        """Ensure that the inputs are valid and prepare them for fit.
        """
        y = _one_hot_encode(y)
        check_consistent_length(X, y)
        X = check_array(X, accept_sparse='csr')
        check_array(y, ensure_2d=False)
        if set(np.unique(y)) != {0, 1}:
            raise ValueError(
                "The target array must either be a 2D dummy encoded (binary)"
                "array or a 1D array with class labels as array elements."
            )

        # Add the intercept column and compute Lipschitz bound the correct way
        if self.fit_intercept:
            X = _add_intercept_col(X)
            X = check_array(X, accept_sparse='csr')

        if lipschitz is None:
            lipschitz = self._compute_lipschitz(X, y)

        if not self.fit_intercept:
            X = _add_intercept_col(X)
            X = check_array(X, accept_sparse='csr')

        return X, y, lipschitz
