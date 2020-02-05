"""
A sample script for group lasso regression
"""

from group_lasso import GroupLasso
from utils import (
    get_groups_from_group_sizes,
    generate_group_lasso_coefficients,
)
import numpy as np
import matplotlib.pyplot as plt


GroupLasso.LOG_LOSSES = True


if __name__ == "__main__":
    np.random.seed(0)

    group_sizes = [np.random.randint(15, 30) for i in range(50)]
    groups = get_groups_from_group_sizes(group_sizes)
    num_coeffs = sum(group_sizes)
    num_datapoints = 10000
    noise_level = 0.5
    coeff_noise_level = 0.05

    print("Generating data")
    X = np.random.standard_normal((num_datapoints, num_coeffs))
    intercept = 2

    print("Generating coefficients")
    w1 = generate_group_lasso_coefficients(group_sizes)
    w2 = generate_group_lasso_coefficients(group_sizes)
    w = np.hstack((w1, w2))
    w *= np.random.random((len(w), 1)) > 0.4
    w += np.random.randn(*w.shape) * coeff_noise_level

    print("Generating targets")
    y = X @ w
    y += np.random.randn(*y.shape) * noise_level * y
    y += intercept

    gl = GroupLasso(
        groups=groups,
        n_iter=100,
        tol=1e-8,
        l1_reg=0.05,
        group_reg=0.18,
        frobenius_lipschitz=False,
        subsampling_scheme=None,
        fit_intercept=True,
    )
    print("Starting fit")
    gl.fit(X, y)

    for i in range(w.shape[1]):
        plt.figure()
        plt.subplot(211)
        plt.plot(w[:, i], ".", label="True weights")
        plt.plot(gl.coef_[:, i], ".", label="Estimated weights")

        plt.subplot(212)
        plt.plot(w[gl.sparsity_mask, i], ".", label="True weights")
        plt.plot(gl.coef_[gl.sparsity_mask, i], ".", label="Estimated weights")
        plt.legend()

    plt.figure()
    plt.plot([w.min(), w.max()], [gl.coef_.min(), gl.coef_.max()], "gray")
    plt.scatter(w, gl.coef_, s=10)
    plt.ylabel("Learned coefficients")
    plt.xlabel("True coefficients")

    plt.figure()
    plt.plot(gl.losses_)

    print("X shape: {X.shape}".format(X=X))
    print("Transformed X shape: {shape}".format(shape=gl.transform(X).shape))
    print("True intercept: {intercept}".format(intercept=intercept))
    print("Estimated intercept: {intercept}".format(intercept=gl.intercept_))
    plt.show()
