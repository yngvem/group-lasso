# TODO: Test the loss

from copy import deepcopy

import numpy as np
import numpy.linalg as la
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from group_lasso import _group_lasso

np.random.seed(0)


class BaseTestGroupLasso:
    MLFitter = _group_lasso.BaseGroupLasso
    UnregularisedMLFitter = None
    num_rows = 200
    num_cols = 30
    configs = [{"n_iter": 1000, "fit_intercept": False}, {"n_iter": 1000, "fit_intercept": True}]

    def all_configs(self, gl):
        for config in self.configs:
            gl.set_params(**config, tol=0)
            yield gl

    @pytest.fixture
    def gl_no_reg(self):
        return self.MLFitter(l1_reg=0, group_reg=0, groups=[])

    @pytest.fixture
    def sklearn_no_reg(self):
        return self.UnregularisedMLFitter()

    @pytest.fixture
    def ml_problem(self):
        raise NotImplementedError

    def random_weights(self):
        return np.random.standard_normal((self.num_cols, 1))

    def test_lipschits(self, gl_no_reg, ml_problem):
        X, y, w = ml_problem

        for gl in self.all_configs(gl_no_reg):
            gl._init_fit(X, y)
            L = gl._compute_lipschitz(X, y)

            g1 = gl._grad(X, y, w)
            for i in range(100):
                w2 = self.random_weights() * i

                g2 = gl._grad(X, y, w2)

                assert la.norm(g1 - g2) <= L * la.norm(w - w2)

    def test_grad(self, gl_no_reg, ml_problem):
        X, y, w = ml_problem
        shape = w.shape
        wrav = w.ravel()
        eps = 1e-5

        for gl in self.all_configs(gl_no_reg):
            gl._init_fit(X, y)
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
        print(gl_no_reg)
        for gl in self.all_configs(gl_no_reg):
            yhat1 = gl.fit_predict(X, y)
            sklearn_no_reg.fit(X, y)
            yhat2 = sklearn_no_reg.predict(X).reshape(yhat1.shape)

            assert np.allclose(yhat1, yhat2)

    def test_high_group_sparsity_yields_zero_coefficients(
        self, ml_problem
    ):
        X, y, w = ml_problem
        reg = 10000
        gl_reg = self.MLFitter(group_reg=reg)
        for gl in self.all_configs(gl_reg):
            gl.fit(X, y)
            assert np.allclose(gl.coef_, 0)

    def test_high_group_sparsity_yields_zero_coefficients(
        self, ml_problem
    ):
        X, y, w = ml_problem
        groups = np.random.randint(0, 10, w.shape)
        reg = 10000
        gl_reg = self.MLFitter(group_reg=reg, l1_reg=0, groups=groups)
        for gl in self.all_configs(gl_reg):
            gl.fit(X, y)
            assert np.allclose(gl.coef_, 0)

    def test_high_l1_sparsity_yields_zero_coefficients(
        self, ml_problem
    ):
        X, y, w = ml_problem
        groups = np.random.randint(0, 10, w.shape)
        reg = 10000
        gl_reg = self.MLFitter(group_reg=0, l1_reg=reg, groups=groups)
        for gl in self.all_configs(gl_reg):
            gl.fit(X, y)
            assert np.allclose(gl.coef_, 0)

    def test_negative_groups_are_ignored(
        self, ml_problem
    ):
        X, y, w = ml_problem
        groups = np.random.randint(-1, 10, w.shape)
        reg = 10000
        gl_reg = self.MLFitter(group_reg=reg, l1_reg=0, groups=groups)
        for gl in self.all_configs(gl_reg):
            gl.fit(X, y)
            nonzero = gl.coef_[groups == -1]
            zero = gl.coef_[groups != -1]

            assert not np.any(np.abs(nonzero) < 1e-10)
            assert np.allclose(zero, 0)

    def test_fit_transform_equals_fit_and_transform(
        self, ml_problem
    ):
        X, y, w = ml_problem
        groups = np.random.randint(0, 10, w.shape)
        reg = 0.01
        gl_reg = self.MLFitter(group_reg=reg, l1_reg=reg, groups=groups)

        for gl in self.all_configs(gl_reg):
            gl_copy = deepcopy(gl)
            yhat1 = gl.fit_transform(X, y)
            gl_copy.fit(X, y)
            yhat2 = gl_copy.transform(X)

            assert yhat1.shape == yhat2.shape
            assert np.allclose(yhat1, yhat2)

    def test_fit_transform_yields_empty_with_high_reg(
        self, ml_problem
    ):
        X, y, w = ml_problem
        groups = np.random.randint(0, 10, w.shape)
        reg = 10000
        gl_reg = self.MLFitter(group_reg=reg, l1_reg=reg, groups=groups)

        for gl in self.all_configs(gl_reg):
            X2 = gl.fit_transform(X, y)
            assert X2.ravel().shape == (0,)


class TestGroupLasso(BaseTestGroupLasso):
    MLFitter = _group_lasso.GroupLasso
    UnregularisedMLFitter = LinearRegression

    @pytest.fixture
    def ml_problem(self):
        np.random.seed(0)
        X = np.random.standard_normal((self.num_rows, self.num_cols))
        w = self.random_weights()
        y = X @ w
        return X, y, w


class TestLogisticGroupLasso(BaseTestGroupLasso):
    MLFitter = _group_lasso.LogisticGroupLasso
    UnregularisedMLFitter = LogisticRegression

    @pytest.fixture
    def ml_problem(self):
        np.random.seed(0)
        X = np.random.standard_normal((self.num_rows, self.num_cols))
        w = self.random_weights()
        y = _group_lasso._sigmoid(X @ w) > 0.5
        return X, y, w

    def test_unregularised_fit_equal_sklearn(
        self, gl_no_reg, sklearn_no_reg, ml_problem
    ):
        X, y, w = ml_problem
        for gl in self.all_configs(gl_no_reg):
            yhat1 = gl.fit_predict(X, y)
            sklearn_no_reg.fit(X, y)
            yhat2 = sklearn_no_reg.predict(X)

            assert np.mean(yhat1.astype(float) - yhat2.astype(float)) < 5e-2


class TestMultinomialGroupLasso(BaseTestGroupLasso):
    MLFitter = _group_lasso.MultinomialGroupLasso
    UnregularisedMLFitter = LogisticRegression
    num_classes = 5

    def random_weights(self):
        return np.random.standard_normal((self.num_cols, self.num_classes))

    @pytest.fixture
    def ml_problem(self):
        np.random.seed(0)
        X = np.random.standard_normal((self.num_rows, self.num_cols))
        w = self.random_weights()
        y = np.argmax(_group_lasso._sigmoid(X @ w), axis=1)
        return X, y, w

    def test_unregularised_fit_equal_sklearn(
        self, gl_no_reg, sklearn_no_reg, ml_problem
    ):
        X, y, w = ml_problem
        sklearn_no_reg.set_params(multi_class="multinomial", solver="lbfgs")
        for gl in self.all_configs(gl_no_reg):
            yhat1 = gl.fit_predict(X, y)
            sklearn_no_reg.fit(X, y[:, np.newaxis])
            yhat2 = sklearn_no_reg.predict(X)

            assert np.mean(yhat1 != yhat2) < 5e-2
