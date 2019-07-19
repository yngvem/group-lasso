from group_lasso import GroupLasso
from utils import (
    get_groups_from_group_sizes,
    generate_group_lasso_coefficients,
)
import numpy as np


# group_lasso._singular_values._DEBUG = True
# group_lasso._group_lasso._DEBUG = True


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.random.seed(0)

    group_sizes = [np.random.randint(15, 30) for i in range(50)]
    groups = get_groups_from_group_sizes(group_sizes)
    num_coeffs = sum(group_sizes)
    num_datapoints = 100_000
    noise_level = 1
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
        n_iter=10,
        tol=1e-8,
        l1_reg=0.02,
        l2_reg=0.05,
        frobenius_lipschitz=True,
        subsampling_scheme=1,
        fit_intercept=True,
    )
    print("Starting fit")
    gl.fit(X, y)

    for i in range(w.shape[1]):
        plt.figure()
        plt.plot(w[:, i], ".", label="True weights")
        plt.plot(gl.coef_[:, i], ".", label="Estimated weights")
        plt.legend()

        plt.figure()
        plt.plot(w[gl.sparsity_mask, i], ".", label="True weights")
        plt.plot(gl.coef_[gl.sparsity_mask, i], ".", label="Estimated weights")
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

    print(f"X shape: {X.shape}")
    print(f"Transformed X shape: {gl.transform(X).shape}")
    print(f"True intercept: {intercept}")
    print(f"Estimated intercept: {gl.intercept_}")
    plt.show()
