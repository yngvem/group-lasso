"""
Some utilities to create gorup lasso coefficients.
"""

import numpy as np


def generate_group_lasso_coefficients(
    group_sizes, inclusion_probability=0.5, coeff_std=1
):
    coefficients = []
    for group_size in group_sizes:
        coefficients_ = np.random.randn(group_size, 1) * coeff_std
        coefficients_ *= np.random.uniform(0, 1) < inclusion_probability
        coefficients.append(coefficients_)

    return np.concatenate(coefficients, axis=0)


def get_groups_from_group_sizes(group_sizes):
    groups_indices = (0, *np.cumsum(group_sizes))
    groups = np.zeros(groups_indices[-1])
    # groups = [None] * len(group_sizes)

    for i, (start, stop) in enumerate(
        zip(groups_indices[:-1], groups_indices[1:])
    ):
        groups[start:stop] = i
        # groups[i] = (start, stop)
    return groups
