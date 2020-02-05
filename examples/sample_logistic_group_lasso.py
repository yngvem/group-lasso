"""
A sample script that runs group lasso for logistic regression.
"""

from group_lasso import LogisticGroupLasso
from utils import (
    get_groups_from_group_sizes,
    generate_group_lasso_coefficients,
)
import group_lasso._singular_values
import group_lasso._group_lasso
import numpy as np


group_lasso._singular_values._DEBUG = True
group_lasso._group_lasso._DEBUG = True
LogisticGroupLasso.LOG_LOSSES = True


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.random.seed(0)

    group_sizes = [np.random.randint(5, 15) for i in range(50)]
    groups = get_groups_from_group_sizes(group_sizes)
    num_coeffs = sum(group_sizes)
    num_datapoints = 100000
    noise_level = 1
    coeff_noise_level = 0.05

    print("Generating data")
    X = np.random.randn(num_datapoints, num_coeffs)
    intercept = 2

    print("Generating coefficients")
    w1 = generate_group_lasso_coefficients(group_sizes)
    w2 = generate_group_lasso_coefficients(group_sizes)
    w = np.hstack((w1, w2))
    w += np.random.randn(*w.shape) * coeff_noise_level

    print("Generating logits")
    y = X @ w
    y += np.random.randn(*y.shape) * noise_level * y
    y += intercept

    print("Generating targets")
    p = 1 / (1 + np.exp(-y))
    z = np.random.binomial(1, p)

    print("Starting fit")
    gl = LogisticGroupLasso(
        groups=groups,
        n_iter=100,
        tol=1e-8,
        group_reg=1e-3,
        l1_reg=1e-3,
        subsampling_scheme=1,
        fit_intercept=True,
    )
    gl.fit(X, z)

    for i in range(w.shape[1]):
        plt.figure()
        plt.plot(w[:, i], ".", label="True weights")
        plt.plot(gl.coef_[:, i], ".", label="Estimated weights")
        plt.legend()

    for i in range(w.shape[1]):
        plt.figure()
        plt.plot(
            w[:, i] / np.linalg.norm(w[:, i]),
            ".",
            label="Normalised true weights",
        )
        plt.plot(
            gl.coef_[:, i] / np.linalg.norm(gl.coef_[:, i]),
            ".",
            label="Normalised estimated weights",
        )
        plt.legend()

    plt.figure()
    plt.plot(gl.losses_)
    plt.title("Loss curve")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")

    plt.figure()
    plt.plot(np.arange(1, len(gl.losses_)), gl.losses_[1:])
    plt.title("Loss curve, ommitting first iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")

    plt.figure()
    plt.plot([w.min(), w.max()], [gl.coef_.min(), gl.coef_.max()], "gray")
    plt.scatter(w, gl.coef_, s=10)
    plt.ylabel("Learned coefficients")
    plt.xlabel("True coefficients")

    print("X shape: {shape}".format(shape=X.shape))
    print("Transformed X shape: {shape}".format(shape=gl.transform(X).shape))
    print("True intercept: {intercept}".format(intercept=intercept))
    print("Estimated intercept: {intercept}".format(intercept=gl.intercept_))
    print("Accuracy: {accuracy}".format(accuracy=np.mean(z == gl.predict(X))))
    plt.show()
