from functools import wraps

import numpy as np


def _extract_from_singleton_iterable(inputs):
    if len(inputs) == 1:
        return inputs[0]
    return tuple(inputs)


def _get_random_row_idxes(num_rows, subsampling_scheme, random_state):
    if subsampling_scheme is None:
        return range(num_rows)
    elif isinstance(subsampling_scheme, str):
        if subsampling_scheme.lower() == "sqrt":
            num_subsampled_rows = int(np.sqrt(num_rows))
        else:
            raise ValueError("Not valid subsampling scheme")
    elif subsampling_scheme < 1 and subsampling_scheme > 0:
        num_subsampled_rows = int(num_rows * subsampling_scheme)
    elif subsampling_scheme >= 1 and isinstance(subsampling_scheme, int):
        if subsampling_scheme > num_rows:
            raise ValueError(
                "Cannot subsample more rows than there are present"
            )
        num_subsampled_rows = subsampling_scheme
    else:
        raise ValueError("Not valid subsampling scheme")

    inds = random_state.choice(num_rows, num_subsampled_rows, replace=False)
    inds.sort()
    return inds


def subsampling_fraction(num_rows, subsampling_scheme, random_state):
    return (
        len(
            _get_random_row_idxes(
                num_rows, subsampling_scheme, random_state=random_state
            )
        )
        / num_rows
    )


def subsample(subsampling_scheme, *Xs, random_state):
    """Subsample along first (0-th) axis of the Xs arrays.

    Arguments
    ---------
    subsampling_scheme : int, float or str
        How to subsample:
         * int or float == 1 -> no subsampling
         * int > 1 -> that many rows are sampled
         * float < 1 -> the fraction of rows to subsample
         * sqrt -> subsample sqrt(num_rows) rows
    """
    assert len(Xs) > 0
    if subsampling_scheme == 1:
        return _extract_from_singleton_iterable(Xs)

    num_rows = Xs[0].shape[0]
    inds = _get_random_row_idxes(
        num_rows, subsampling_scheme, random_state=random_state
    )
    return _extract_from_singleton_iterable([X[inds, :] for X in Xs])


class Subsampler:
    """
    Utility for subsampling along the first (0-th) axis of the Xs arrays.

    Arguments
    ---------
    num_indices : int
        How many indices the arrays to subsample from have
    subsampling_scheme : int, float or str
        How to subsample:
         * int or float == 1 -> no subsampling
         * int > 1 -> that many rows are sampled
         * float < 1 -> the fraction of rows to subsample
         * sqrt -> subsample sqrt(num_rows) rows
    random_state : np.random.RandomState
    """

    def __init__(self, num_indices, subsampling_scheme, random_state):
        self.random_state = random_state
        self.subsampling_scheme = subsampling_scheme
        self.set_num_indices(num_indices)

    def set_num_indices(self, num_indices):
        self.num_indices_ = num_indices
        self.update_indices()

    def subsample(self, *Xs):
        if self.subsampling_scheme == 1:
            return _extract_from_singleton_iterable(Xs)

        return _extract_from_singleton_iterable(
            [X[self.curr_indices_] for X in Xs]
        )

    def update_indices(self):
        self.curr_indices_ = _get_random_row_idxes(
            self.num_indices_,
            self.subsampling_scheme,
            random_state=self.random_state,
        )

    def subsample_apply(self, f, *full_inputs):
        @wraps(f)
        def new_f(*args, **kwargs):
            subsampled_inputs = self.subsample(*full_inputs)
            return f(*subsampled_inputs, *args, **kwargs)

        return new_f
