from group_lasso import *
import numpy as np


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    np.random.seed(0)

    group_sizes = [np.random.randint(3, 10) for i in range(50)]
    groups = get_groups_from_group_sizes(group_sizes)
    num_coeffs = sum(group_sizes)
    num_datapoints = 1_000_000
    noise_level = 0.5

    print('Generating data')
    X = np.random.randn(num_datapoints, num_coeffs)
    print('Generating coefficients')
    w = generate_group_lasso_coefficients(group_sizes, noise_level=0.05)

    print('Generating targets')
    y = X@w
    y += np.random.randn(*y.shape)*noise_level*y

    gl = GroupLassoRegressor(
        groups=groups, n_iter=50, tol=0.01, reg=0.1, subsampling_scheme=0.001
    )
    print('Starting fit')
    gl.fit(X, y)

    plt.plot(w, '.', label='True weights')
    plt.plot(gl.coef_, '.', label='Estimated weights')
    plt.legend()
    plt.show()