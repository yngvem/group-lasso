"""
"""
import numpy as np


def extract_ohe_groups(onehot_encoder):
    """Extract a vector with group indices from a scikit-learn OneHotEncoder

    Arguments
    ---------
    onehot_encoder : sklearn.preprocessing.OneHotEncoder

    Returns
    -------
    np.ndarray
        A group-vector that can be used with the group lasso regularised
        linear models.
    """
    if not hasattr(onehot_encoder, "categories_"):
        raise ValueError(
            "Cannot extract group labels from an unfitted OneHotEncoder instance."
        )

    categories = onehot_encoder.categories_
    return np.concatenate(
        [
            group * np.ones_like(category)
            for group, category in enumerate(categories)
        ]
    )
