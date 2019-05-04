import numpy as np


def generate_group_lasso_coefficients(
    group_sizes,
    inclusion_probability=0.5,
    included_std=1,
    noise_level=0,
):
    coefficients = []
    for group_size in group_sizes:
        coefficients_ = np.random.randn(group_size, 1)*included_std
        coefficients_ *= (np.random.uniform(0, 1) < inclusion_probability)
        coefficients_ += np.random.randn(group_size, 1)*noise_level
        coefficients.append(coefficients_)

    return np.concatenate(coefficients, axis=0)


def get_groups_from_group_sizes(group_sizes):
    groups = (0, *np.cumsum(group_sizes))
    return list(zip(groups[:-1], groups[1:]))
