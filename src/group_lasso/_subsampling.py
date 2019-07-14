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

    num_rows = len(Xs[0])
    inds = _get_random_row_idxes(
        num_rows, subsampling_scheme, random_state=random_state
    )
    return _extract_from_singleton_iterable([X[inds, :] for X in Xs])
