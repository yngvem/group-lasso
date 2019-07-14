from contextlib import contextmanager
import numpy as np
import numpy.linalg as la
import pytest
from group_lasso import _group_lasso
from sklearn.linear_model import LinearRegression, LogisticRegression

np.random.seed(0)


class BaseTestGroupLasso:
    MLFitter = _group_lasso.BaseGroupLasso
    UnregularisedMLFitter = None
    num_rows = 200
    num_cols = 30
    configs = [{"n_iter": 1000}]

    @contextmanager
    def all_configs(self, gl):
        for config in self.configs:
            gl.set_params(**config, tol=0)
            yield gl

    @pytest.fixture
    def gl_no_reg(self):
        return self.MLFitter(reg=0, groups=[])

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

        with self.all_configs(gl_no_reg) as gl:
            gl._init_fit(X, y)
            L = gl._compute_lipschitz(X, y)

            g1 = gl._grad(X, y, w)
            for i in range(100):
                w2 = self.random_weights() * i

                g2 = gl._grad(X, y, w2)

                assert la.norm(g1 - g2) <= L * la.norm(w - w2)

    def test_grad(self, gl_no_reg, ml_problem):
        X, y, w = ml_problem
        w = self.random_weights()
        eps = 1e-5

        with self.all_configs(gl_no_reg) as gl:
            gl._init_fit(X, y)
            loss = gl._unregularised_loss(X, y, w)
            dw = np.empty_like(w)
            g = gl._grad(X, y, w)
            for i, _ in enumerate(w):
                w_ = w.copy()
                w_[i] += eps
                dw[i] = (gl._unregularised_loss(X, y, w_) - loss) / (
                    w_[i] - w[i]
                )
                assert abs((dw[i] - g[i])) / abs(g[i]) < 1e-2
            assert la.norm(dw - g) / la.norm(g) < 1e-2

    def test_unregularised_fit_equal_sklearn(
        self, gl_no_reg, sklearn_no_reg, ml_problem
    ):
        X, y, w = ml_problem
        with self.all_configs(gl_no_reg) as gl:
            yhat1 = gl.fit_predict(X, y)
            sklearn_no_reg.fit(X, y)
            yhat2 = sklearn_no_reg.predict(X).reshape(yhat1.shape)

            assert np.allclose(yhat1, yhat2)

    # TODO: FIND SPARSITY PATTERNS WITH NOISE


class TestGroupLasso(BaseTestGroupLasso):
    MLFitter = _group_lasso.GroupLasso
    UnregularisedMLFitter = LinearRegression

    @pytest.fixture
    def ml_problem(self):
        X = np.random.standard_normal((self.num_rows, self.num_cols))
        w = self.random_weights()
        y = X @ w
        return X, y, w


class TestLogisticGroupLasso(BaseTestGroupLasso):
    MLFitter = _group_lasso.LogisticGroupLasso
    UnregularisedMLFitter = LogisticRegression

    @pytest.fixture
    def ml_problem(self):
        X = np.random.standard_normal((self.num_rows, self.num_cols))
        w = self.random_weights()
        y = _group_lasso._sigmoid(X @ w) > 0.5
        return X, y, w

    def test_unregularised_fit_equal_sklearn(
        self, gl_no_reg, sklearn_no_reg, ml_problem
    ):
        X, y, w = ml_problem
        with self.all_configs(gl_no_reg) as gl:
            yhat1 = gl.fit_predict(X, y)
            sklearn_no_reg.fit(X, y)
            yhat2 = sklearn_no_reg.predict(X)

            assert np.mean(yhat1.astype(float) - yhat2.astype(float)) < 5e-2
