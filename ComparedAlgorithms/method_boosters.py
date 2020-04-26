# -*- coding: utf-8 -*-
"""
method_boosters.py - Boosters for least-square solvers
======================================================

This module contains all available boosters for least-square methods, the Cholesky-booster and the Caratheodory-booster.
See the following example of using any booster, i.e cholesky-booster.

Example:
    boosted_solver = cholesky_booster(existing_solver)

"""

import numpy as np
from numpy.linalg import cholesky
from Infrastructure.utils import Callable, List, Scalar, ColumnVector, Vector, Matrix, ex, compute_serial
from ComparedAlgorithms.basic_caratheodory_set import caratheodory_set_python as _caratheodory_set_python
from ComparedAlgorithms.base_least_square_solver import BaseSolver
from ComparedAlgorithms.basic_caratheodory_set import cluster_average


_cholesky_covariance: Callable = lambda all_data_mat, clusters_count: cholesky(all_data_mat.T.dot(all_data_mat)).T


@ex.capture
def _LMS_coreset(all_data_mat: Matrix, cross_validation_folds: int, clusters_count: int) -> Matrix:
    sub_matrices: List[Matrix] = np.array_split(all_data_mat, cross_validation_folds)
    coreset: Matrix = np.vstack([create_coreset_fast_caratheodory(sub_mat, clusters_count) for sub_mat in sub_matrices])
    return coreset


def __booster_template(coreset_action: Callable, booster_name: str = "") -> Callable:
    def __booster(existing_regression_method: Callable, perform_normalization: bool = False) -> Callable:

        class _BoostedSolver(BaseSolver):
            @ex.capture
            def __init__(self, data_features: Matrix, output_samples: ColumnVector, n_alphas: int,
                         cross_validation_folds: int, _rnd):
                super(_BoostedSolver, self).__init__(data_features, output_samples, n_alphas, cross_validation_folds)
                self._inner_model = None
                data_length, data_dim = self._data_features.shape
                self._beta: Scalar = self._cross_validation_folds * (data_dim + 1) ** 2 + self._cross_validation_folds
                self._beta = np.sqrt(self._beta / data_length)
                self._perform_normalization = perform_normalization

            def fit(self):
                all_data_mat: Matrix = np.hstack((self._data_features, self._output_samples.reshape(-1, 1)))
                all_data_coreset: Matrix = coreset_action(all_data_mat, self._cross_validation_folds)
                L: Matrix = all_data_coreset[:, :-1]
                new_output_samples: ColumnVector = all_data_coreset[:, -1]

                if self._perform_normalization:
                    L *= self._beta
                    new_output_samples *= self._beta

                self._inner_model = existing_regression_method(L, new_output_samples, self._n_alphas,
                                                               self._cross_validation_folds)

                self._fitted_coefficients = self._inner_model.fit()
                return self._fitted_coefficients

        _BoostedSolver.__name__ = f'{booster_name}{existing_regression_method.__name__}'
        return _BoostedSolver
    return __booster


cholesky_booster: Callable = __booster_template(_cholesky_covariance, booster_name="Cholesky-boosted")
caratheodory_booster: Callable = __booster_template(_LMS_coreset, booster_name="Caratheodory-boosted")


# def caratheodory_set_python(points: Matrix, weights: ColumnVector) -> (Matrix, ColumnVector):
#     """
#     This method computes Caratheodory subset for the columns of a :math:`dxn` Matrix, with given weights.
#
#     Args:
#         points(Matrix): A :math:`dxn` Matrix.
#         weights(ColumnVector): A weights for the columns of :math:`A`.
#
#     Returns:
#         A Caratheodory subset of :math:`d^{2]+1` columns of :math:`A, as a :math:`dx(d^{2}+1)` Matrix,
#         and their weights as a ColumnVector.
#     """
#     dim: int = len(points)  # The dimension of all rows, or rows of the matrix, d.
#     remaining_indices = np.arange(0, points.shape[1])
#
#     while len(remaining_indices) > dim + 1:
#         diff_points: Matrix = points[:, remaining_indices]
#         diff_points = (diff_points[:, 1:].T - diff_points[:, 0].T).T
#         _, _, V = np.linalg.svd(diff_points, full_matrices=True)
#         v = V[-1]
#         v = np.insert(v, [0], -np.sum(v))  # The last num_points - dim - 2 zeroes are removed.
#         div: Vector = np.empty_like(v)
#         div.fill(np.inf)
#         # alpha: Scalar = np.inf
#         np.divide(weights[remaining_indices], v, out=div, where=v > 0)
#         alpha: Scalar = min(div)
#         # for i in range(len(v)):
#         #     if v[i] > 0 and weights[remaining_indices[i]] / v[i] < alpha:
#         #         alpha = weights[remaining_indices[i]] / v[i]
#         weights[remaining_indices] -= alpha * v
#         remaining_indices = np.nonzero(weights > 1e-15)[0]
#
#     return weights[remaining_indices], remaining_indices


def _greedy_split(arr: Vector, n: int, axis: int = 0, default_block_size: int = 1) -> (List[Vector], Vector):
    """Greedily splits an array into n blocks.

    Splits array arr along axis into n blocks such that:
        - blocks 1 through n-1 are all the same size
        - the sum of all block sizes is equal to arr.shape[axis]
        - the last block is nonempty, and not bigger than the other blocks

    Intuitively, this "greedily" splits the array along the axis by making
    the first blocks as big as possible, then putting the leftovers in the
    last block.
    """
    length: int = arr.shape[axis]

    # compute the size of each of the first n-1 blocks
    if length < n:
        n = default_block_size
    block_size: int = np.floor(length / float(n))

    # the indices at which the splits will occur
    ix: Vector = np.arange(block_size, length, block_size, dtype=int)
    return np.split(arr, ix, axis), ix


def fast_caratheodory_set_python(points: Matrix, weights: ColumnVector, accuracy_time_tradeoff_const: int,
                                 cluster_inner_indices=None) -> (Matrix, ColumnVector):
    """
    This method computes Caratheodory subset for the columns of a :math:`dxn` Matrix, with given weights, using the
    fast Caratheodory subset algorithm with a given number of clusters.

    Args:
        points(Matrix): A :math:`dxn` Matrix.
        weights(ColumnVector): A weights for the columns of :math:`A`.
        accuracy_time_tradeoff_const(int): The number of clusters to use for computing the Caratheodory subset.
        cluster_inner_indices:

    Returns:
        A Caratheodory subset of :math:`d^{2]+1` columns of :math:`A, as a :math:`dx(d^{2}+1)` Matrix,
        and their weights as a ColumnVector.
    """
    points_num: int = points.shape[1]  # The number of points, or columns of the matrix, :math:`n`.
    dim: int = len(points)  # The dimension of all rows, or rows of the matrix, :math:`d`.

    if points_num <= dim + 1:
        returned_indices = cluster_inner_indices
        if cluster_inner_indices is None:
            returned_indices = [x.tolist() for x in np.arange(points_num)]
        return weights, returned_indices

    # Split points into :math:`k` clusters and find the weighted means of every cluster.
    clusters, split_index = _greedy_split(points, accuracy_time_tradeoff_const, axis=1, default_block_size=dim + 2)
    split_weights = np.split(weights, split_index)
    if cluster_inner_indices is None:
        cluster_inner_indices = np.arange(points_num)
    split_indices = np.split(cluster_inner_indices, split_index)
    arguments_to_parallel_mean = list(zip(clusters, split_weights))
    means_and_clusters_weights = compute_serial(cluster_average, arguments_to_parallel_mean)

    clusters_means = np.asfortranarray(np.vstack([result[0] for result in means_and_clusters_weights]))
    clusters_weights = np.array([result[1] for result in means_and_clusters_weights])

    # Perform caratheodory method on the set of means and the total weights of each cluster.
    coreset_weights, chosen_indices = _caratheodory_set_python(clusters_means.T, clusters_weights)

    C = []
    c_weights = []
    all_chosen_indices = []
    # Take C as the union of rows from clusters which the caratheodory set consists of.
    for cluster_total_weight, cluster_index in zip(coreset_weights, chosen_indices):
        cluster = clusters[cluster_index]
        cluster_weights = split_weights[cluster_index]
        C.append(cluster)
        all_chosen_indices += np.ravel(split_indices[cluster_index]).tolist()
        c_weights.append(cluster_total_weight * cluster_weights / np.sum(cluster_weights))

    # Assign new weights for each row in C.
    # Perform fast caratheodory method recursively.
    C = np.hstack(C)
    c_weights = np.hstack(c_weights)
    new_weights, all_chosen_indices = fast_caratheodory_set_python(C, c_weights, accuracy_time_tradeoff_const,
                                                                   all_chosen_indices)
    return new_weights, all_chosen_indices


def create_coreset_fast_caratheodory(A: Matrix, clusters_count: int) -> Matrix:
    """
    This method computes the outer-products of the rows of the input matrix. Then it computes a coreset for it,
    using :func:`fast_caratheodory_set_python` with ``clusters_count`` clusters.

    Args:
        A(Matrix): An input :math:nxd matrix.
        clusters_count(int): The number of clusters to use for computing the coreset.

    Returns:
        A :math:`(d^{2}+1)xd` Matrix coreset for the input matrix.
    """
    n: int = A.shape[0]  # The number of points, or rows of the matrix, n.

    rows_outer_products: Matrix = np.einsum("ij,ik->ijk", A, A, optimize=True)
    rows_outer_products = np.ascontiguousarray(rows_outer_products.reshape((n, -1))).T
    weights, remaining_indices = fast_caratheodory_set_python(rows_outer_products, np.ones(n) / n, clusters_count)
    reduced_mat: Matrix = np.multiply(A[remaining_indices, :].T, np.sqrt(n * weights)).T
    return reduced_mat
