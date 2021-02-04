from copy import deepcopy
from itertools import product
import pickle

import numpy as np
import numpy.linalg as la
import pytest
from scipy import sparse
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.estimator_checks import parametrize_with_checks, check_estimator

from group_lasso import BaseGroupLasso, GroupLasso, LogisticGroupLasso
import group_lasso._group_lasso as _group_lasso

np.random.seed(0)


@pytest.mark.parametrize("reg", np.logspace(-3, 1, 10))
def test_l1_prox(reg):
    x = np.random.randn(100)
    y = _group_lasso._l1_prox(x, reg)
    for xi, yi in zip(x, y):
        if np.abs(xi) > reg:
            assert np.abs(yi) == pytest.approx(np.abs(xi) - reg)
        else:
            assert yi == 0


@pytest.mark.parametrize(
    ("reg", "n_per_group"), product(np.logspace(-3, 1, 10), [2, 3, 5])
)
def test_group_l2_prox(reg, n_per_group):
    x = np.random.randn(50 * n_per_group)
    groups = [np.zeros(x.shape, dtype=np.bool) for _ in range(50)]
    for i, group in enumerate(groups):
        group[i * n_per_group : (i + 1) * n_per_group] = True
    y = _group_lasso._group_l2_prox(x, [reg] * len(groups), groups)
    for group in groups:
        xi = x[group]
        yi = y[group]
        if np.linalg.norm(xi) > reg:
            assert np.linalg.norm(yi) == pytest.approx(
                np.linalg.norm(xi) - reg
            )
        else:
            assert np.linalg.norm(yi) == pytest.approx(0)


@pytest.mark.parametrize(
    ("template, output"),
    [
        (None, -1),
        (3, 3),
        (-1, -1),
        ([1, 2, 3], [1, 2, 3]),
        ([1, 1, 1], [1, 1, 1]),
        ([-1, -1, -1], [-1, -1, -1]),
        ([None, None, None], [-1, -1, -1]),
        ([None, -1, 1], [-1, -1, 1]),
        ([None, 1, 2], [-1, 1, 2]),
        ([[None, -1, -1]], [[-1, -1, -1]]),
        ([[None, 1, 2]], [[-1, 1, 2]]),
        ([[1, 2, 3]], [[1, 2, 3]]),
        ([[1, 1, 1]], [[1, 1, 1]]),
        ([[-1, -1, -1]], [[-1, -1, -1]]),
        ([[None, 1, 2], [1, 2, 3]], [[-1, 1, 2], [1, 2, 3]]),
    ]
)
def test_parse_group_iterable(template, output):
    assert _group_lasso._parse_group_iterable(template) == output

class BaseTestGroupLasso:
    MLFitter = _group_lasso.BaseGroupLasso
    UnregularisedMLFitter = None
    num_rows = 200
    num_cols = 30
    configs = [
        {"n_iter": 1000, "fit_intercept": False},
        {"n_iter": 1000, "fit_intercept": True},
    ]

    def all_configs(self, gl):
        for config in self.configs:
            gl.set_params(**config, tol=0)
            yield gl

    @pytest.fixture
    def gl_no_reg(self):
        return self.MLFitter(
            l1_reg=0, group_reg=0, groups=[], supress_warning=True
        )

    @pytest.fixture
    def sklearn_no_reg(self):
        return self.UnregularisedMLFitter()

    @pytest.fixture
    def ml_problem(self):
        raise NotImplementedError

    @pytest.fixture
    def sparse_ml_problem(self):
        raise NotImplementedError

    def test_sklearn_compat(self):
        check_estimator(self.MLFitter)

    def random_weights(self):
        return np.random.standard_normal((self.num_cols, 1))

    def test_regularisation_is_scaled_correctly(self):
        groups = [1, 1, 1, 2, 2, 3]
        gl = self.MLFitter(
            group_reg=1, groups=groups, n_iter=2, supress_warning=True
        )
        X = np.random.randn(100, 6)
        y = np.random.randint(0, 2, (100, 2))
        gl.scale_reg = "group_size"
        gl._init_fit(X, y, 1)
        assert gl.group_reg_vector_ == pytest.approx(
            [np.sqrt(3), np.sqrt(2), 1]
        )
        gl.scale_reg = None
        gl._init_fit(X, y, 1)
        assert gl.group_reg_vector_ == pytest.approx([1] * 3)
        gl.scale_reg = "inverse_group_size"
        gl._init_fit(X, y, 1)
        assert gl.group_reg_vector_ == pytest.approx(
            [1 / np.sqrt(3), 1 / np.sqrt(2), 1]
        )

    def test_intercept_used_correctly_for_loss(self, ml_problem):
        X, y, w = ml_problem
        groups = np.random.randint(1, 5, size=w.shape)
        gl = self.MLFitter(
            group_reg=1,
            groups=groups,
            fit_intercept=True,
            supress_warning=True,
        )

        gl.fit(X, y)
        w_hat = _group_lasso._join_intercept(gl.intercept_, gl.coef_)
        X_intercept = _group_lasso._add_intercept_col(X)
        assert gl.loss(X, y) == gl._loss(X_intercept, y, w_hat)

    def test_reg_is_correct(self, gl_no_reg, ml_problem):
        X, y, w = ml_problem
        gl = gl_no_reg
        gl._init_fit(X, y, lipschitz=None)
        assert gl._regulariser(w) == 0

        reg = 0.1
        gl.l1_reg = reg
        w2 = np.concatenate((w[0:1, :] * 0, w))
        assert gl._regulariser(w2) == np.linalg.norm(w.ravel(), 1) * reg

        gl.groups = np.arange(len(w.ravel())).reshape(w.shape)
        gl.group_reg = reg
        gl._init_fit(X, y, lipschitz=None)
        assert gl._regulariser(w2) == pytest.approx(
            2 * np.linalg.norm(w.ravel(), 1) * reg
        )

        gl.groups = np.concatenate(
            [np.arange(len(w.ravel()) / 2)] * 2
        ).reshape(w.shape)
        gl.group_reg = reg
        gl._init_fit(X, y, lipschitz=None)
        regulariser = 0
        for group in np.unique(gl.groups):
            regulariser += (
                reg * np.sqrt(2) * np.linalg.norm(w[gl.groups == group])
            )
        assert gl._regulariser(w2) == pytest.approx(
            regulariser + np.linalg.norm(w.ravel(), 1) * reg
        )

    def test_grad(self, gl_no_reg, ml_problem):
        X, y, w = ml_problem
        shape = w.shape
        wrav = w.ravel()
        eps = 1e-5

        for gl in self.all_configs(gl_no_reg):
            gl._init_fit(X, y, lipschitz=None)
            loss = gl._unregularised_loss(X, y, w)
            dw = np.empty_like(wrav)
            g = gl._grad(X, y, w)
            g = g.ravel()
            for i, _ in enumerate(wrav):
                wrav_ = wrav.copy()
                wrav_[i] += eps
                w_ = wrav_.reshape(shape)
                dw[i] = (gl._unregularised_loss(X, y, w_) - loss) / (
                    wrav_[i] - wrav[i]
                )
            assert np.allclose(dw, g, rtol=1e-2, atol=1e-5)

    def test_unregularised_fit_equal_sklearn(
        self, gl_no_reg, sklearn_no_reg, ml_problem
    ):
        X, y, w = ml_problem
        for gl in self.all_configs(gl_no_reg):
            yhat1 = gl.fit_predict(X, y)
            sklearn_no_reg.fit(X, y)
            yhat2 = sklearn_no_reg.predict(X).reshape(yhat1.shape)

            th = 0.01
            pred_diff = yhat1.astype(float) - yhat2.astype(float)
            if np.linalg.norm(pred_diff, 1) / y.shape[0] > th:
                diff_gl = np.linalg.norm(yhat1.astype(float) - y.astype(float))
                diff_sk = np.linalg.norm(yhat2.astype(float) - y.astype(float))
                assert diff_gl < diff_sk

    def test_fitted_is_picklable(
        self, gl_no_reg, ml_problem
    ):  
        X, y, w = ml_problem
        gl = gl_no_reg
        gl.warm_start = True
        gl.group_reg = 100
        try:
            pickle.dumps(gl)
        except pickle.PicklingError:
            assert False, "Cannot pickle unfitted estimator"
        gl.fit(X, y)
        try:
            pickle.dumps(gl)
        except pickle.PicklingError:
            assert False, "Cannot pickle fitted estimator"

    def test_warm_start_is_possible(self, gl_no_reg, ml_problem):
        X, y, w = ml_problem
        gl = gl_no_reg
        gl.warm_start = True
        gl.group_reg = 100
        gl.fit(X, y)
        coef = gl.coef_.copy()
        gl.group_reg = 0
        gl.fit(X, y)
        coef2 = gl.coef_.copy()
        assert not np.allclose(coef, coef2)

    def test_unregularised_sparse_fit_equal_sklearn(
        self, gl_no_reg, sklearn_no_reg, sparse_ml_problem
    ):
        X, y, w = sparse_ml_problem
        sklearn_no_reg.n_iter = 1000
        sklearn_no_reg.tol = 1e-10
        for gl in self.all_configs(gl_no_reg):
            yhat1 = gl.fit_predict(X, y)
            sklearn_no_reg.fit(X, y)
            yhat2 = sklearn_no_reg.predict(X).reshape(yhat1.shape)

            th = 0.01
            pred_diff = yhat1.astype(float) - yhat2.astype(float)
            if np.linalg.norm(pred_diff, 1) / y.shape[0] > th:
                diff_gl = np.linalg.norm(yhat1.astype(float) - y.astype(float))
                diff_sk = np.linalg.norm(yhat2.astype(float) - y.astype(float))
                assert diff_gl < diff_sk

    def test_high_group_sparsity_yields_zero_coefficients(self, ml_problem):
        X, y, w = ml_problem
        reg = 10000
        gl_reg = self.MLFitter(group_reg=reg, supress_warning=True)
        for gl in self.all_configs(gl_reg):
            gl.fit(X, y)
            np.testing.assert_allclose(gl.coef_, 0)

    def test_high_group_sparsity_yields_zero_coefficients(self, ml_problem):
        X, y, w = ml_problem
        groups = np.random.randint(0, 10, w.shape)
        reg = 10000
        gl_reg = self.MLFitter(
            group_reg=reg, l1_reg=0, groups=groups, supress_warning=True
        )
        for gl in self.all_configs(gl_reg):
            gl.fit(X, y)
            np.testing.assert_allclose(gl.coef_, 0)

    def test_high_l1_sparsity_yields_zero_coefficients(self, ml_problem):
        X, y, w = ml_problem
        groups = np.random.randint(0, 10, w.shape)
        reg = 10000
        gl_reg = self.MLFitter(
            group_reg=0, l1_reg=reg, groups=groups, supress_warning=True
        )
        for gl in self.all_configs(gl_reg):
            gl.fit(X, y)
            np.testing.assert_allclose(gl.coef_, 0)

    def test_negative_groups_are_ignored(self, ml_problem):
        X, y, w = ml_problem
        groups = np.random.randint(-1, 10, w.shape)
        reg = 10000
        gl_reg = self.MLFitter(
            group_reg=reg, l1_reg=0, groups=groups, supress_warning=True
        )
        for gl in self.all_configs(gl_reg):
            gl.fit(X, y)
            nonzero = gl.coef_[groups == -1]
            zero = gl.coef_[groups != -1]

            assert not np.any(np.abs(nonzero) < 1e-10)
            np.testing.assert_allclose(zero, 0)

    def test_fit_transform_equals_fit_and_transform(self, ml_problem):
        X, y, w = ml_problem
        groups = np.random.randint(0, 10, w.shape)
        reg = 0.01
        gl_reg = self.MLFitter(
            group_reg=reg, l1_reg=reg, groups=groups, supress_warning=True
        )

        for gl in self.all_configs(gl_reg):
            gl_copy = deepcopy(gl)
            yhat1 = gl.fit_transform(X, y)
            gl_copy.fit(X, y)
            yhat2 = gl_copy.transform(X)

            assert yhat1.shape == yhat2.shape
            assert np.allclose(yhat1, yhat2)

    def test_fit_transform_yields_empty_with_high_reg(self, ml_problem):
        X, y, w = ml_problem
        groups = np.random.randint(0, 10, w.shape)
        reg = 10000
        gl_reg = self.MLFitter(
            group_reg=reg, l1_reg=reg, groups=groups, supress_warning=True
        )

        for gl in self.all_configs(gl_reg):
            X2 = gl.fit_transform(X, y)
            assert X2.ravel().shape == (0,)

    @pytest.mark.parametrize(
        ("reg", "multitarget_groups"),
        product(np.logspace(-5, 2, 8), [True, False]),
    )
    def test_chosen_groups_is_correct(
        self, ml_problem, reg, multitarget_groups
    ):
        X, y, w = ml_problem
        if multitarget_groups:
            groups = np.random.randint(0, 10, w.shape)
        else:
            groups = np.random.randint(0, 10, w.shape[0])
        gl = self.MLFitter(group_reg=reg, groups=groups, supress_warning=True)
        gl.fit(X, y)

        chosen_groups = set(gl.chosen_groups_)
        for group in np.unique(groups):
            mask = groups == group

            if group in chosen_groups:
                assert np.linalg.norm(gl.coef_[mask]) > 1e-8
            else:
                assert np.linalg.norm(gl.coef_[mask]) <= 1e-8

    def test_changing_intercept_changes_prediction(
        self, ml_problem, gl_no_reg
    ):
        X, y, w = ml_problem
        gl_no_reg.fit(X, y)
        pred = gl_no_reg.predict(X)
        gl_no_reg.intercept_ *= 1e10 * np.random.standard_normal(
            gl_no_reg.intercept_.shape
        )
        assert not np.allclose(pred, gl_no_reg.predict(X))


class TestGroupLasso(BaseTestGroupLasso):
    MLFitter = GroupLasso
    UnregularisedMLFitter = LinearRegression

    @pytest.fixture
    def ml_problem(self):
        np.random.seed(0)
        X = np.random.standard_normal((self.num_rows, self.num_cols))
        w = self.random_weights()
        y = X @ w
        return X, y, w

    @pytest.fixture
    def sparse_ml_problem(self):
        X = sparse.random(self.num_rows, self.num_cols, random_state=0)
        w = self.random_weights()
        y = X @ w
        return X, y, w


class TestLogisticGroupLasso(BaseTestGroupLasso):
    MLFitter = LogisticGroupLasso
    UnregularisedMLFitter = LogisticRegression
    num_classes = 5

    def random_weights(self):
        return np.random.standard_normal((self.num_cols, self.num_classes))

    @pytest.fixture
    def ml_problem(self):
        np.random.seed(0)
        X = np.random.standard_normal((self.num_rows, self.num_cols))
        w = self.random_weights()
        y = np.argmax(_group_lasso._softmax(X @ w), axis=1)
        y = LabelBinarizer().fit_transform(y)
        return X, y, w

    @pytest.fixture
    def sparse_ml_problem(self):
        X = sparse.random(self.num_rows, self.num_cols, random_state=0)
        X = sparse.dok_matrix(X)
        for row in range(self.num_rows):
            col = np.random.randint(self.num_cols)
            X[row, col] = np.random.standard_normal()
        w = self.random_weights()
        y = np.argmax(_group_lasso._softmax(X @ w), axis=1)
        y = LabelBinarizer().fit_transform(y)
        return X, y, w

    def test_unregularised_fit_equal_sklearn(
        self, gl_no_reg, sklearn_no_reg, ml_problem
    ):
        X, y, w = ml_problem
        sklearn_no_reg.set_params(multi_class="multinomial", solver="lbfgs")
        for gl in self.all_configs(gl_no_reg):
            yhat1 = gl.fit_predict(X, np.argmax(y, axis=1))
            sklearn_no_reg.fit(X, np.argmax(y, axis=1))
            yhat2 = sklearn_no_reg.predict(X)

            assert np.mean(yhat1 != yhat2) < 5e-2

    def test_unregularised_sparse_fit_equal_sklearn(
        self, gl_no_reg, sklearn_no_reg, sparse_ml_problem
    ):
        X, y, w = sparse_ml_problem
        sklearn_no_reg.n_iter = 1000
        sklearn_no_reg.tol = 1e-10
        for gl in self.all_configs(gl_no_reg):
            yhat1 = gl.fit_predict(X, np.argmax(y, axis=1)[:, np.newaxis])
            sklearn_no_reg.fit(X, np.argmax(y, axis=1)[:, np.newaxis])
            yhat2 = sklearn_no_reg.predict(X).reshape(yhat1.shape)

            yhat1 = gl._encode(yhat1)
            yhat2 = gl._encode(yhat2)
            th = 0.01
            pred_diff = yhat1.astype(float) - yhat2.astype(float)
            if np.linalg.norm(pred_diff, 1) / y.shape[0] > th:
                diff_gl = np.linalg.norm(yhat1.astype(float) - y.astype(float))
                diff_sk = np.linalg.norm(yhat2.astype(float) - y.astype(float))
                assert diff_gl < diff_sk

    # TODO: This is a copy from base because parametrize didn't work with this subclass.
    # TODO: Submit issue to pytest repo
    @pytest.mark.parametrize(
        ("reg", "multitarget_groups"),
        product(np.logspace(-5, 2, 8), [True, False]),
    )
    def test_chosen_groups_is_correct(
        self, ml_problem, reg, multitarget_groups
    ):
        X, y, w = ml_problem
        if multitarget_groups:
            groups = np.random.randint(0, 10, w.shape)
        else:
            groups = np.random.randint(0, 10, w.shape[0])
        gl = self.MLFitter(group_reg=reg, groups=groups, supress_warning=True)
        gl.fit(X, y)

        chosen_groups = set(gl.chosen_groups_)
        for group in np.unique(groups):
            mask = groups == group

            if group in chosen_groups:
                assert np.linalg.norm(gl.coef_[mask]) > 1e-8
            else:
                assert np.linalg.norm(gl.coef_[mask]) <= 1e-8
