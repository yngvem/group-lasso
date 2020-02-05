"""
A sample script for multinomial group lasso.
"""

from group_lasso import MultinomialGroupLasso
from utils import (
    get_groups_from_group_sizes,
    generate_group_lasso_coefficients,
)
import group_lasso._singular_values
import group_lasso._group_lasso
import numpy as np


group_lasso._singular_values._DEBUG = True
group_lasso._group_lasso._DEBUG = True


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.random.seed(0)

    group_sizes = [np.random.randint(5, 15) for i in range(50)]
    groups = get_groups_from_group_sizes(group_sizes)
    num_coeffs = sum(group_sizes)
    num_datapoints = 10000
    num_classes = 5
    noise_level = 1
    coeff_noise_level = 0.05

    print("Generating data")
    X = np.random.randn(num_datapoints, num_coeffs)
    intercept = np.arange(num_classes) * 10

    print("Generating coefficients")
    w = np.random.randn(num_coeffs, num_classes)
    for group in np.unique(groups):
        w[groups == group, :] *= np.random.random() > 0.3
    w += np.random.randn(*w.shape) * coeff_noise_level

    print("Generating logits")
    y = X @ w
    y += (
        np.random.randn(*y.shape)
        * noise_level
        / np.linalg.norm(y, axis=1, keepdims=True)
    )
    y += intercept

    print("Generating targets")
    p = np.exp(y) / (np.exp(y).sum(1, keepdims=True))
    z = [np.random.choice(np.arange(num_classes), p=pi) for pi in p]
    z = np.array(z)

    print("Starting fit")
    gl = MultinomialGroupLasso(
        groups=groups,
        n_iter=10,
        tol=1e-8,
        group_reg=5e-3,
        l1_reg=1e-4,
        fit_intercept=True,
    )
    gl.fit(X, z)

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
        plt.title("Normalised coefficients")
        plt.legend()

    plt.figure()
    plt.plot([w.min(), w.max()], [gl.coef_.min(), gl.coef_.max()], "gray")
    plt.scatter(w, gl.coef_, s=10)
    plt.ylabel("Learned coefficients")
    plt.xlabel("True coefficients")

    print("X shape: {shape}".format(shape=X.shape))
    print("Transformed X shape: {shape}".format(shape=gl.transform(X).shape))
    print(
        "True intercept: {intercept}".format(
            intercept=(intercept - intercept.mean())
            / np.linalg.norm(intercept - intercept.mean())
        )
    )
    print(
        "Estimated intercept: {intercept}".format(
            intercept=(gl.intercept_ - gl.intercept_.mean())
            / np.linalg.norm(gl.intercept_ - gl.intercept_.mean())
        )
    )
    print("Accuracy: {accuracy}".format(accuracy=np.mean(z == gl.predict(X))))
    plt.show()
