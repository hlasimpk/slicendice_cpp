import os
import numpy as np
import json
from scipy import sparse
import warnings

input_file = os.path.join(os.getcwd(), 'prog', 'data.json')
f = open(input_file)
data = json.load(f)

input_array = []
for i in data.keys():
    coords = []
    for j in data[i].keys():
        coords.append(data[i][j])
    input_array.append(coords)


def stable_cumsum(arr, axis=None, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum.

    Warns if the final cumulative sum does not match the sum (up to the chosen
    tolerance).

    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat.
    axis : int, default=None
        Axis along which the cumulative sum is computed.
        The default (None) is to compute the cumsum over the flattened array.
    rtol : float, default=1e-05
        Relative tolerance, see ``np.allclose``.
    atol : float, default=1e-08
        Absolute tolerance, see ``np.allclose``.

    Returns
    -------
    out : ndarray
        Array with the cumulative sums along the chosen axis.
    """
    out = np.cumsum(arr, axis=axis, dtype=np.float64)
    expected = np.sum(arr, axis=axis, dtype=np.float64)
    if not np.all(
        np.isclose(
            out.take(-1, axis=axis), expected, rtol=rtol, atol=atol, equal_nan=True
        )
    ):
        warnings.warn(
            (
                "cumsum was found to be unstable: "
                "its last element does not correspond to sum"
            ),
            RuntimeWarning,
        )
    return out

def safe_sparse_dot(a, b, *, dense_output=False):
    """Dot product that handle the sparse matrix case correctly.

    Parameters
    ----------
    a : {ndarray, sparse matrix}
    b : {ndarray, sparse matrix}
    dense_output : bool, default=False
        When False, ``a`` and ``b`` both being sparse will yield sparse output.
        When True, output will always be a dense array.

    Returns
    -------
    dot_product : {ndarray, sparse matrix}
        Sparse if ``a`` and ``b`` are sparse and ``dense_output=False``.
    """
    if a.ndim > 2 or b.ndim > 2:
        if sparse.issparse(a):
            # sparse is always 2D. Implies b is 3D+
            # [i, j] @ [k, ..., l, m, n] -> [i, k, ..., l, n]
            b_ = np.rollaxis(b, -2)
            b_2d = b_.reshape((b.shape[-2], -1))
            ret = a @ b_2d
            ret = ret.reshape(a.shape[0], *b_.shape[1:])
        elif sparse.issparse(b):
            # sparse is always 2D. Implies a is 3D+
            # [k, ..., l, m] @ [i, j] -> [k, ..., l, j]
            a_2d = a.reshape(-1, a.shape[-1])
            ret = a_2d @ b
            ret = ret.reshape(*a.shape[:-1], b.shape[1])
        else:
            ret = np.dot(a, b)
    else:
        ret = a @ b

    if (
        sparse.issparse(a)
        and sparse.issparse(b)
        and dense_output
        and hasattr(ret, "toarray")
    ):
        return ret.toarray()
    return ret

def row_norms(X, squared=False):
    """Row-wise (squared) Euclidean norm of X.

    Equivalent to np.sqrt((X * X).sum(axis=1)), but also supports sparse
    matrices and does not create an X.shape-sized temporary.

    Performs no input validation.

    Parameters
    ----------
    X : array-like
        The input array.
    squared : bool, default=False
        If True, return squared norms.

    Returns
    -------
    array-like
        The row-wise (squared) Euclidean norm of X.
    """
    if sparse.issparse(X):
        X = X.tocsr()
        norms = csr_row_norms(X)
    else:
        norms = np.einsum("ij,ij->i", X, X)

    if not squared:
        np.sqrt(norms, norms)
    return norms

def _euclidean_distances(X, Y, X_norm_squared=None, Y_norm_squared=None, squared=False):
    """Computational part of euclidean_distances

    Assumes inputs are already checked.

    If norms are passed as float32, they are unused. If arrays are passed as
    float32, norms needs to be recomputed on upcast chunks.
    TODO: use a float64 accumulator in row_norms to avoid the latter.
    """
    if X_norm_squared is not None:
        if X_norm_squared.dtype == np.float32:
            XX = None
        else:
            XX = X_norm_squared.reshape(-1, 1)
    elif X.dtype == np.float32:
        XX = None
    else:
        XX = row_norms(X, squared=True)[:, np.newaxis]

    if Y is X:
        YY = None if XX is None else XX.T
    else:
        if Y_norm_squared is not None:
            if Y_norm_squared.dtype == np.float32:
                YY = None
            else:
                YY = Y_norm_squared.reshape(1, -1)
        elif Y.dtype == np.float32:
            YY = None
        else:
            YY = row_norms(Y, squared=True)[np.newaxis, :]

    distances = -2 * safe_sparse_dot(X, Y.T, dense_output=True)

    distances += XX
    distances += YY

    np.maximum(distances, 0, out=distances)

    # Ensure that distances between vectors and themselves are set to 0.0.
    # This may not be the case due to floating point rounding errors.
    if X is Y:
        np.fill_diagonal(distances, 0)

    return distances if squared else np.sqrt(distances, out=distances)

def _kmeans_plus_plus(X, n_clusters, x_squared_norms, sample_weight, random_state, n_local_trials=None):
    n_samples, n_features = X.shape

    sample_weight = np.ones(n_samples, dtype=X.dtype)

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    n_local_trials = 2 + int(np.log(n_clusters))

    random_state = np.random.mtrand._rand
    center_id = random_state.choice(n_samples, p=sample_weight / sample_weight.sum())
    center_id = 46 # hardcoding for testing

    indices = np.full(n_clusters, -1, dtype=int)

    centers[0] = X[center_id]

    indices[0] = center_id

    closest_dist_sq = _euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms, squared=True
    )
    current_pot = closest_dist_sq @ sample_weight

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.uniform(size=n_local_trials) * current_pot
        rand_vals = [16265.23297954, 19865.80160536] # hardcoding for testing

        candidate_ids = np.searchsorted(
            stable_cumsum(sample_weight * closest_dist_sq), rand_vals
        )


        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = _euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True
        )


        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)


        candidates_pot = distance_to_candidates @ sample_weight.reshape(-1, 1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        if sparse.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]
        indices[c] = best_candidate
    return centers, indices

X = np.array(input_array)
print(_kmeans_plus_plus(X, 2, None, None, None, None))


variances = np.var(X, axis=0)
print(np.mean(variances))

X_mean = X.mean(axis=0)
X -= X_mean
print(X)

center_distances = _euclidean_distances(X, X)
print(center_distances)
print(center_distances / 2)