import warnings
from abc import ABC, abstractmethod
from math import sqrt
from numbers import Number

import numpy as np
import numpy.linalg as la
from scipy import sparse
from scipy.special import logsumexp
from sklearn.base import (BaseEstimator, ClassifierMixin, RegressorMixin,
                          TransformerMixin)
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import (check_array, check_consistent_length,
                           check_random_state)
from sklearn.utils.multiclass import unique_labels
from sklearn.exceptions import NotFittedError

from group_lasso._fista import FISTAProblem
from group_lasso._singular_values import find_largest_singular_value
from group_lasso._subsampling import Subsampler

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
    num_classes = w.shape[1]
    return np.concatenate([np.array(b).reshape(1, num_classes), w], axis=0)


def _add_intercept_col(X):
    ones = np.ones([X.shape[0], 1])
    if sparse.issparse(X):
        return sparse.hstack((ones, X))
    return np.hstack([ones, X])


def _parse_group_iterable(iterable_or_number):
	try:
		iter(iterable_or_number)
	except TypeError:
		if iterable_or_number is None:
			return -1
		else:
			return iterable_or_number
	else:
		return [_parse_group_iterable(i) for i in iterable_or_number]


class BaseGroupLasso(ABC, BaseEstimator, TransformerMixin):
    """Base class for sparse group lasso regularised optimisation.

    This class implements the Sparse Group Lasso [1] regularisation for
    optimisation problems with Lipschitz continuous gradients, which is
    approximately equivalent to having a bounded second derivative.

    The loss is optimised using the FISTA algorithm proposed in [2] with the
    generalised gradient-based restarting scheme proposed in [3].

    Parameters
    ----------
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
    scale_reg : str [in {"group_size", "none", "inverse_group_size"] or None
        How to scale the group-wise regularisation coefficients. In the
        original group lasso paper scaled the regularisation by the square
        root of the elements in each group so that each variable has the
        same effect on the regularisation. This is not sensible for dummy
        encoded variables, as these always have either unit or zero norm.
        ``scale_reg`` should therefore be None if all variables are dummy
        variables. Finally, if the group size shouldn't be considered when
        choosing variables, then inverse_group_size should be used instead
        as that divide by the square root of the group size, removing the
        dependence of group size on the regularisation strength.
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
        groups=None,
        group_reg=0.05,
        l1_reg=0.00,
        n_iter=100,
        tol=1e-5,
        scale_reg="group_size",
        subsampling_scheme=None,
        fit_intercept=True,
        random_state=None,
        warm_start=False,
        old_regularisation=False,
        supress_warning=False,
    ):
        self.groups = groups
        self.group_reg = group_reg
        self.scale_reg = scale_reg
        self.l1_reg = l1_reg
        self.n_iter = n_iter
        self.tol = tol
        self.subsampling_scheme = subsampling_scheme
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.old_regularisation = old_regularisation
        self.warm_start = warm_start
        self.supress_warning = supress_warning

    def _more_tags(self):
        return {'multioutput': True}

    def _regulariser(self, w):
        """The regularisation penalty for a given coefficient vector, ``w``.

        The first element of the coefficient vector is the intercept which
        is sliced away.
        """
        regulariser = 0
        coef_ = _split_intercept(w)[1]
        for group, reg in zip(self.groups_, self.group_reg_vector_):
            regulariser += reg * la.norm(coef_[group])
        regulariser += self.l1_reg * la.norm(coef_.ravel(), 1)
        return regulariser

    def _get_reg_strength(self, group, reg):
        """Get the regularisation coefficient for one group.
        """
        scale_reg = str(self.scale_reg).lower()
        if scale_reg == "group_size":
            scale = sqrt(group.sum())
        elif scale_reg == "none":
            scale = 1
        elif scale_reg == "inverse_group_size":
            scale = 1 / sqrt(group.sum())
        else:
            raise ValueError(
                '``scale_reg`` must be equal to "group_size",'
                ' "inverse_group_size" or "none"'
            )
        return reg * scale

    def _get_reg_vector(self, reg):
        """Get the group-wise regularisation coefficients from ``reg``.
        """
        if isinstance(reg, Number):
            reg = [
                self._get_reg_strength(group, reg) for group in self.groups_
            ]
        else:
            reg = list(reg)
        return reg

    @abstractmethod
    def _unregularised_loss(self, X_aug, y, w):  # pragma: nocover
        """The unregularised reconstruction loss.
        """
        raise NotImplementedError

    def _loss(self, X, y, w):
        """The group-lasso regularised loss.

        Parameters
        ----------
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

        Parameters
        ----------
        X : np.ndarray
            Data matrix, ``X.shape == (num_datapoints, num_features)``
        y : np.ndarray
            Target vector/matrix, ``y.shape == (num_datapoints, num_targets)``,
            or ``y.shape == (num_datapoints,)``
        """
        X_aug = _add_intercept_col(X)
        w = _join_intercept(self.intercept_, self.coef_)
        return self._loss(X_aug, y, w)

    @abstractmethod
    def _estimate_lipschitz(self, X_aug, y):  # pragma: nocover
        """Compute Lipschitz bound for the gradient of the unregularised loss.

        The Lipschitz bound is with respect to the coefficient vector or
        matrix.
        """
        raise NotImplementedError

    @abstractmethod
    def _grad(self, X_aug, y, w):  # pragma: nocover
        """Compute the gradient of the unregularised loss wrt the coefficients.
        """
        raise NotImplementedError

    def _unregularised_gradient(self, X_aug, y, w):
        g = self._grad(X_aug, y, w)
        if not self.fit_intercept:
            g[0] = 0
        return g

    def _scaled_prox(self, w, lipschitz):
        """Apply the proximal map of the scaled regulariser to ``w``.

        The scaling is the inverse lipschitz coefficient.
        """
        b, w_ = _split_intercept(w)
        l1_reg = self.l1_reg
        group_reg_vector = self.group_reg_vector_
        if not self.old_regularisation:
            l1_reg = l1_reg / lipschitz
            group_reg_vector = np.asarray(group_reg_vector) / lipschitz

        w_ = _l1_l2_prox(w_, l1_reg, group_reg_vector, self.groups_)
        return _join_intercept(b, w_)

    def _minimise_loss(self):
        """Use the FISTA algorithm to solve the group lasso regularised loss.
        """
        # Need transition period before the correct regulariser is used without warning
        def callback(x, it_num, previous_x=None):
            X_, y_ = self.subsampler_.subsample(self.X_aug_, self.y_)
            self.subsampler_.update_indices()
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

                grad_norm = la.norm(
                    self._unregularised_gradient(self.X_aug_, self.y_, w)
                )
                print("\tWeight norm: {wnorm}".format(wnorm=la.norm(w)))
                print("\tGrad: {gnorm}".format(gnorm=grad_norm))
                print(
                    "\tRelative grad: {relnorm}".format(
                        relnorm=grad_norm / la.norm(w)
                    )
                )
                print(
                    "\tLipschitz: {lipschitz}".format(
                        lipschitz=optimiser.lipschitz
                    )
                )

        weights = _join_intercept(self.intercept_, self.coef_)
        optimiser = FISTAProblem(
            self.subsampler_.subsample_apply(
                self._unregularised_loss, self.X_aug_, self.y_
            ),
            self._regulariser,
            self.subsampler_.subsample_apply(
                self._unregularised_gradient, self.X_aug_, self.y_
            ),
            self._scaled_prox,
            self.lipschitz_,
        )
        weights = optimiser.minimise(
            weights, n_iter=self.n_iter, tol=self.tol, callback=callback
        )
        self.lipschitz_ = optimiser.lipschitz
        self.intercept_, self.coef_ = _split_intercept(weights)

    def _check_valid_parameters(self):
        """Check that the input parameters are valid.
        """
        assert all(reg >= 0 for reg in self.group_reg_vector_)
        groups = self.group_ids_
        assert len(self.group_reg_vector_) == len(
            np.unique(groups[groups >= 0])
        )
        assert self.n_iter > 0
        assert self.tol >= 0

    def _prepare_dataset(self, X, y, lipschitz):
        """Ensure that the inputs are valid and prepare them for fit.
        """
        X = check_array(X, accept_sparse="csc")
        y = check_array(y, ensure_2d=False)

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        # Center X for numerical stability
        if not sparse.issparse(X) and self.fit_intercept:
            X_means = X.mean(axis=0, keepdims=True)
            X = X - X_means
        else:
            X_means = np.zeros((1, X.shape[1]))

        # Add the intercept column and compute Lipschitz bound the correct way
        if self.fit_intercept:
            X = _add_intercept_col(X)
            X = check_array(X, accept_sparse="csc")

        if lipschitz is None:
            lipschitz = self._estimate_lipschitz(X, y)

        if not self.fit_intercept:
            X = _add_intercept_col(X)
            X = check_array(X, accept_sparse="csc")

        return X, X_means, y, lipschitz

    def _init_fit(self, X, y, lipschitz):
        """Initialise model and check inputs.
        """
        self.random_state_ = check_random_state(self.random_state)

        check_consistent_length(X, y)
        X, X_means, y, lipschitz = self._prepare_dataset(X, y, lipschitz)
        
        self.subsampler_ = Subsampler(
            X.shape[0], self.subsampling_scheme, self.random_state_
        )

        groups = self.groups
        if groups is None:
            groups = np.arange(X.shape[1]-1)

        self.group_ids_ = np.array(_parse_group_iterable(groups))

        self.groups_ = [
            self.group_ids_ == u
            for u in np.unique(self.group_ids_) if u >= 0
        ]
        self.group_reg_vector_ = self._get_reg_vector(self.group_reg)

        self.losses_ = []

        if not self.warm_start or not hasattr(self, "coef_"):
            self.coef_ = np.zeros((X.shape[1] - 1, y.shape[1]))
            self.intercept_ = np.zeros((1, self.coef_.shape[1]))

        self._check_valid_parameters()
        self.X_aug_, self.y_, self.lipschitz_ = X, y, lipschitz
        self._X_means_ = X_means
        if not self.old_regularisation and not self.supress_warning:
            warnings.warn(_OLD_REG_WARNING)

    def fit(self, X, y, lipschitz=None):
        """Fit a group-lasso regularised linear model.
        """
        self._init_fit(X, y, lipschitz=lipschitz)
        self._minimise_loss()
        self.intercept_ -= (self._X_means_ @ self.coef_).reshape(
            self.intercept_.shape
        )
        return self

    def _compute_scores(self, X):
        w = _join_intercept(self.intercept_, self.coef_)
        if X.shape[1] == self.coef_.shape[0]:
            X = _add_intercept_col(X)

        return X @ w

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
        warnings.warn(
            "This property is discontinued, use sparsity_mask_ instead of sparsity_mask."
        )
        return self.sparsity_mask_

    def _get_chosen_coef_mask(self, coef_):
        mean_abs_coef = abs(coef_.mean())
        return np.abs(coef_) > 1e-10 * mean_abs_coef

    @property
    def sparsity_mask_(self):
        """A boolean mask indicating whether features are used in prediction.
        """
        coef_ = self.coef_.mean(1)
        return self._get_chosen_coef_mask(coef_)

    @property
    def chosen_groups_(self):
        """A set of the coosen group ids.
        """
        groups = self.group_ids_
        if groups.ndim == 1:
            sparsity_mask = self.sparsity_mask_
        else:
            sparsity_mask = self._get_chosen_coef_mask(self.coef_).ravel()
        groups = groups.ravel()
        # TODO: Add regression test with list input for groups

        return set(np.unique(groups[sparsity_mask]))

    def transform(self, X):
        """Remove columns corresponding to zero-valued coefficients.
        """
        if not hasattr(self, 'coef_'):
            raise NotFittedError
        X = check_array(X, accept_sparse="csc")
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                "The transformer {} does not raise an error when the number of "
                "features in transform is different from the number of features in "
                "fit.".format(self.__class__.__name__)
            )

        return X[:, self.sparsity_mask_]

    def fit_transform(self, X, y, lipschitz=None):
        """Fit a group lasso model to X and y and remove unused columns from X
        """
        self.fit(X, y, lipschitz)
        return self.transform(X)


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

    Parameters
    ----------
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
    scale_reg : str [in {"group_size", "none", "inverse_group_size"] or None
        How to scale the group-wise regularisation coefficients. In the
        original group lasso paper scaled the regularisation by the square
        root of the elements in each group so that each variable has the
        same effect on the regularisation. This is not sensible for dummy
        encoded variables, as these always have either unit or zero norm.
        ``scale_reg`` should therefore be None if all variables are dummy
        variables. Finally, if the group size shouldn't be considered when
        choosing variables, then inverse_group_size should be used instead
        as that divide by the square root of the group size, removing the
        dependence of group size on the regularisation strength.
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
        scale_reg="group_size",
        subsampling_scheme=None,
        fit_intercept=True,
        frobenius_lipschitz=False,
        random_state=None,
        warm_start=False,
        old_regularisation=False,
        supress_warning=False,
    ):
        super().__init__(
            groups=groups,
            l1_reg=l1_reg,
            group_reg=group_reg,
            n_iter=n_iter,
            tol=tol,
            scale_reg=scale_reg,
            subsampling_scheme=subsampling_scheme,
            fit_intercept=fit_intercept,
            random_state=random_state,
            warm_start=warm_start,
            old_regularisation=old_regularisation,
            supress_warning=supress_warning,
        )
        self.frobenius_lipschitz = frobenius_lipschitz

    def fit(self, X, y, lipschitz=None):
        """Fit a group lasso regularised linear regression model.

        Parameters
        ----------
        X : np.ndarray
            Data matrix
        y : np.ndarray
            Target vector or matrix
        lipschitz : float or None [default=None]
            A Lipshitz bound for the mean squared loss with the given
            data and target matrices. If None, this is estimated.
        """
        return super().fit(X, y, lipschitz=lipschitz)

    def predict(self, X):
        """Predict using the linear model.
        """
        if not hasattr(self, 'coef_'):
            raise NotFittedError
        X = check_array(X, accept_sparse="csc")
        scores = self._compute_scores(X)
        if scores.ndim == 2 and scores.shape[1] == 1:
            return scores.reshape(scores.shape[0])
        return scores

    def _unregularised_loss(self, X_aug, y, w):
        MSE = np.sum((X_aug @ w - y) ** 2) / X_aug.shape[0]
        return 0.5 * MSE

    def _grad(self, X_aug, y, w):
        SSE_grad = _l2_grad(X_aug, y, w)
        return SSE_grad / X_aug.shape[0]

    def _estimate_lipschitz(self, X_aug, y):
        num_rows = X_aug.shape[0]
        if self.frobenius_lipschitz:
            if sparse.issparse(X_aug):
                return sparse.linalg.norm(X_aug, "fro") ** 2 / num_rows
            return la.norm(X_aug, "fro") ** 2 / num_rows

        s_max = find_largest_singular_value(
            X_aug,
            subsampling_scheme=self.subsampling_scheme,
            random_state=self.random_state_,
        )
        SSE_lipschitz = 1.5 * s_max ** 2
        return SSE_lipschitz / num_rows


def _softmax(logit):
    logit = logit - logit.max(1, keepdims=True)
    expl = np.exp(logit)
    return expl / expl.sum(axis=1, keepdims=True)


def _softmax_cross_entropy(X, Y, W):
    logit = X @ W
    logit -= logit.max(axis=1, keepdims=True)  # To prevent overflow
    return -np.sum(Y * (logit - logsumexp(logit, axis=1, keepdims=True)))
    # -np.sum(Y * np.log(_softmax(X@W)))
    # -np.sum(Y * np.log(np.exp(X@W) / np.sum(np.exp(X@W), axis=1, keepdims=True)))
    # -np.sum(Y * (np.log(np.exp(X@W)) - np.log(np.sum(np.exp(X@W), axis=1, keepdims=True))))
    # -np.sum(Y * (X@W - logsumexp(X@W, axis=1, keepdims=True)))


class LogisticGroupLasso(BaseGroupLasso, ClassifierMixin):
    """Sparse group lasso regularised multi-class logistic regression.

    This class implements the Sparse Group Lasso [1] regularisation for
    multi-class logistic regression with a cross entropy penalty.

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

    Parameters
    ----------
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
    scale_reg : str [in {"group_size", "none", "inverse_group_size"] or None
        How to scale the group-wise regularisation coefficients. In the
        original group lasso paper scaled the regularisation by the square
        root of the elements in each group so that each variable has the
        same effect on the regularisation. This is not sensible for dummy
        encoded variables, as these always have either unit or zero norm.
        ``scale_reg`` should therefore be None if all variables are dummy
        variables. Finally, if the group size shouldn't be considered when
        choosing variables, then inverse_group_size should be used instead
        as that divide by the square root of the group size, removing the
        dependence of group size on the regularisation strength.
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
        scale_reg="group_size",
        subsampling_scheme=None,
        fit_intercept=True,
        random_state=None,
        warm_start=False,
        old_regularisation=False,
        supress_warning=False,
    ):
        if subsampling_scheme is not None:
            warnings.warn(
                "Subsampling is not stable for logistic regression group lasso."
            )
        super().__init__(
            groups=groups,
            group_reg=group_reg,
            l1_reg=l1_reg,
            n_iter=n_iter,
            tol=tol,
            scale_reg=scale_reg,
            subsampling_scheme=subsampling_scheme,
            fit_intercept=fit_intercept,
            random_state=random_state,
            warm_start=warm_start,
            old_regularisation=old_regularisation,
            supress_warning=supress_warning,
        )
    
    def _more_tags(self):
        return {'multiclass': True}

    def _unregularised_loss(self, X_aug, y, w):
        return _softmax_cross_entropy(X_aug, y, w).sum() / X_aug.shape[0]

    def _grad(self, X_aug, y, w):
        p = _softmax(X_aug @ w)

        return X_aug.T @ (p - y) / X_aug.shape[0]

    def _estimate_lipschitz(self, X_aug, y):
        if sparse.issparse(X_aug):
            return sparse.linalg.norm(X_aug, "fro")
        else:
            return la.norm(X_aug, "fro")

    def predict_proba(self, X):
        if not hasattr(self, 'coef_'):
            raise NotFittedError
        X = check_array(X, accept_sparse="csc")
        scores = self._compute_scores(X)
        return _softmax(scores)

    def predict(self, X):
        """Predict using the linear model.
        """
        proba = self.predict_proba(X)
        if proba.shape[1] == 2:
            proba = proba[:, 1]
        return self.label_binarizer_.inverse_transform(proba)

    def _encode(self, y):
        """One-hot encoding for the labels.
        """
        y = self.label_binarizer_.transform(y)
        if np.asarray(y).shape[1] == 1:
            ones = np.ones((y.shape[0], 1))
            y = np.hstack(((ones - y.sum(1, keepdims=True)), y,))
        return y

    def _prepare_dataset(self, X, y, lipschitz):
        """Ensure that the inputs are valid and prepare them for fit.
        """
        self.classes_ = unique_labels(y)
        self.label_binarizer_ = LabelBinarizer()
        self.label_binarizer_.fit(y)
        y = self._encode(y)
        check_consistent_length(X, y)
        X = check_array(X, accept_sparse="csc")
        check_array(y, ensure_2d=False)

        if set(np.unique(y)) != {0, 1}:
            raise ValueError(
                "The target array must either be a 2D dummy encoded (binary)"
                "array or a 1D array with class labels as array elements."
            )

        # Center X for numerical stability
        if not sparse.issparse(X) and self.fit_intercept:
            X_means = X.mean(axis=0, keepdims=True)
            X = X - X_means
        else:
            X_means = np.zeros((1, X.shape[1]))

        # Add the intercept column and compute Lipschitz bound the correct way
        if self.fit_intercept:
            X = _add_intercept_col(X)
            X = check_array(X, accept_sparse="csc")

        if lipschitz is None:
            lipschitz = np.abs(X).max(1).mean()

        if not self.fit_intercept:
            X = _add_intercept_col(X)
            X = check_array(X, accept_sparse="csc")

        return X, X_means, y, lipschitz
