import numpy as np
import pytest
from sklearn.preprocessing import OneHotEncoder

from group_lasso import utils


def test_one_hot_encoder_groups():
    X = np.hstack(
        (np.random.randint(0, 3, (100, 1)), np.random.randint(0, 2, (100, 1)))
    )
    ohe = OneHotEncoder()
    ohe.fit(X)
    groups = utils.extract_ohe_groups(ohe)
    assert list(groups) == [0, 0, 0, 1, 1]


def test_extract_ohe_groups_fails_on_unfitted_ohe():
    ohe = OneHotEncoder()
    with pytest.raises(ValueError):
        utils.extract_ohe_groups(ohe)
