from group_lasso import GroupLasso
from group_lasso._utils import get_groups_from_group_sizes, generate_group_lasso_coefficients
import group_lasso._singular_values
import group_lasso._group_lasso_fista
import numpy as np


group_lasso._singular_values.DEBUG = True
group_lasso._group_lasso_fista.DEBUG = True


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    np.random.seed(0)

    group_sizes = [np.random.randint(3, 10) for i in range(50)]
    groups = get_groups_from_group_sizes(group_sizes)
    num_coeffs = sum(group_sizes)
    num_datapoints = 100_000
    noise_level = 1
    coeff_noise_level = 0.05

    print('Generating data')
    X = np.random.randn(num_datapoints, num_coeffs)
    print('Generating coefficients')
    w = generate_group_lasso_coefficients(group_sizes)
    w += np.random.randn(*w.shape)*coeff_noise_level

    print('Generating targets')
    y = X@w
    y += np.random.randn(*y.shape)*noise_level*y

    gl = GroupLasso(
        groups=groups, n_iter=50, tol=1e-16, reg=0.1, subsampling_scheme=0.1,
    )
    print('Starting fit')
    gl.fit(X, y)

    plt.plot(w, '.', label='True weights')
    plt.plot(gl.coef_, '.', label='Estimated weights')
    plt.legend()

    plt.figure()
    plt.plot(gl._losses)
    plt.show()