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
    configs = [{"n_iter": 1000}]

    def all_configs(self, gl):
        for config in self.configs:
            gl.set_params(**config, tol=0)
            yield gl

    @pytest.fixture
    def gl_no_reg(self):
        return self.MLFitter(l1_reg=0, l2_reg=0, groups=[])

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
        eps = 1e-5

        for gl in self.all_configs(gl_no_reg):
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
                print(
                    f"{dw[i, 0]:.3e}, {g[i, 0]:.3e}, {w[i, 0]:.3e}, "
                    f"{dw[i, 0] - g[i, 0]:.3e}, "
                    f"{(dw[i, 0] - g[i, 0])/g[i, 0]:.3e}"
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

    # TODO: FIND SPARSITY PATTERNS WITH NOISE


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
