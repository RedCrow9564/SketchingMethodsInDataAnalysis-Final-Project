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


_cholesky_covariance: Callable = lambda all_data_mat: cholesky(all_data_mat.T.dot(all_data_mat)).T


@ex.capture
def _LMS_coreset(all_data_mat: Matrix, cross_validation_folds: int, clusters_count: int) -> Matrix:
    sub_matrices: List[Matrix] = np.array_split(all_data_mat, cross_validation_folds)
    coreset: Matrix = np.vstack([create_coreset_fast_caratheodory(sub_mat, clusters_count) for sub_mat in sub_matrices])
    return coreset


def __booster_template(coreset_action: Callable, cross_validation_folds: int, booster_name: str = "") -> Callable:
    def __booster(existing_regression_method: Callable, perform_normalization: bool = False) -> Callable:
        def _boosted_regression_method(data_features: Matrix, output_samples: ColumnVector, n_alphas: int = 100,
                                       calc_residuals: bool = True) -> (ColumnVector, ColumnVector):
            all_data_mat: Matrix = np.hstack((data_features, output_samples.reshape(-1, 1)))
            all_data_coreset: Matrix = coreset_action(all_data_mat)
            L: Matrix = all_data_coreset[:, :-1]
            new_output_samples: ColumnVector = all_data_coreset[:, -1]

            if perform_normalization:
                data_length, data_dim = data_features.shape
                beta: Scalar = cross_validation_folds * (data_dim + 1) ** 2 + cross_validation_folds
                beta = np.sqrt(beta / data_length)
                L *= beta
                new_output_samples *= beta

            coefficients, _ = existing_regression_method(L, new_output_samples, calc_residuals=False, n_alphas=n_alphas)
            residuals: ColumnVector = output_samples - data_features.dot(coefficients) if calc_residuals else -1
            return coefficients, residuals

        _boosted_regression_method.__name__ = f'{booster_name}_{existing_regression_method.__name__}'
        return _boosted_regression_method
    return __booster


cholesky_booster: Callable = __booster_template(_cholesky_covariance, cross_validation_folds=3,
                                                booster_name="Cholesky-boosted")
caratheodory_booster: Callable = __booster_template(_LMS_coreset, cross_validation_folds=3,
                                                    booster_name="Caratheodory-boosted")


def _caratheodory_set_python(points: Matrix, weights: ColumnVector) -> (Matrix, ColumnVector):
    """
    This method computes Caratheodory subset for the columns of a :math:`dxn` Matrix, with given weights.

    Args:
        points(Matrix): A :math:`dxn` Matrix.
        weights(ColumnVector): A weights for the columns of :math:`A`.

    Returns:
        A Caratheodory subset of :math:`d^{2]+1` columns of :math:`A, as a :math:`dx(d^{2}+1)` Matrix,
        and their weights as a ColumnVector.
    """
    dim: int = len(points)  # The dimension of all rows, or rows of the matrix, d.

    while len(weights) > dim + 1:
        diff_points: Matrix = (points[:, 1:].T - points[:, 0].T).T
        _, _, V = np.linalg.svd(diff_points, full_matrices=True)
        v = V[-1]
        v = np.insert(v, [0], -np.sum(v))  # The last num_points - dim - 2 zeroes are removed.
        div: ColumnVector = np.empty_like(v).fill(-np.inf)
        div = np.divide(weights, v, out=div, where=v > 0)
        alpha: Scalar = min(div[div > 0])
        weights -= alpha * v
        points = points[:, weights > 1e-15]
        weights = weights[weights > 1e-15]

    return points, weights


def _greedy_split(arr: Vector, n: int, axis: int = 0, default_block_size: int = 1) -> List[Vector]:
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
    return np.split(arr, ix, axis)


def fast_caratheodory_set_python(points: Matrix, weights: ColumnVector,
                                 accuracy_time_tradeoff_const: int) -> (Matrix, ColumnVector):
    """
    This method computes Caratheodory subset for the columns of a :math:`dxn` Matrix, with given weights, using the
    fast Cartheodory subset algorithm with a given number of clusters.

    Args:
        points(Matrix): A :math:`dxn` Matrix.
        weights(ColumnVector): A weights for the columns of :math:`A`.
        accuracy_time_tradeoff_const(int): The number of clusters to use for computing the Caratheodory subset.

    Returns:
        A Caratheodory subset of :math:`d^{2]+1` columns of :math:`A, as a :math:`dx(d^{2}+1)` Matrix,
        and their weights as a ColumnVector.
    """
    points_num: int = points.shape[1]  # The number of points, or columns of the matrix, :math:`n`.
    dim: int = len(points)  # The dimension of all rows, or rows of the matrix, :math:`d`.

    if points_num <= dim + 1:
        return points, weights

    # Split points into :math:`k` clusters and find the weighted means of every cluster.
    clusters = _greedy_split(points, accuracy_time_tradeoff_const, axis=1, default_block_size=dim + 2)
    split_weights = _greedy_split(weights, accuracy_time_tradeoff_const, default_block_size=dim + 2)
    arguments_to_parallel_mean = [
        (cluster, 1, cluster_weights, True) for cluster, cluster_weights in zip(clusters, split_weights)]
    means_and_clusters_weights = compute_serial(np.average, arguments_to_parallel_mean)

    clusters_means = np.vstack([result[0] for result in means_and_clusters_weights]).T
    clusters_weights = np.hstack([result[1][0] for result in means_and_clusters_weights])

    # Perform caratheodory method on the set of means and the total weights of each cluster.
    means_coreset, coreset_weights = _caratheodory_set_python(clusters_means, clusters_weights)

    C = []
    c_weights = []
    # Take C as the union of rows from clusters which the caratheodory set consists of.
    for cluster, cluster_weights, cluster_mean in zip(clusters, split_weights, clusters_means.T):
        for mean, weight in zip(means_coreset.T, coreset_weights):
            if np.all(mean == cluster_mean):
                C.append(cluster)
                c_weights.append(weight * cluster_weights / np.sum(cluster_weights))
                break

    # Assign new weights for each row in C.
    # Perform fast caratheodory method recursively.
    C = np.hstack(C)
    c_weights = np.hstack(c_weights)
    coreset, new_weights = fast_caratheodory_set_python(C, c_weights, accuracy_time_tradeoff_const)
    return coreset, new_weights


def create_coreset_fast_caratheodory(A: Matrix, clusters_count:int) -> Matrix:
    """
    This method computes the outer-products of the rows of the input matrix. Then it computes a coreset for it,
    using :func:`fast_caratheodory_set_python` with ``clusters_count`` clusters.

    Args:
        A(Matrix): An input :math:nxd matrix.
        clusters_count(int): The number of clusters to use for computing the coreset.

    Returns:
        A :math:`(d^{2}+1)xn` Matrix coreset for the input matrix.
    """
    n: int = A.shape[0]  # The number of points, or rows of the matrix, n.
    d: int = A.shape[1]  # The dimension of all rows, or columns of the matrix, d.

    rows_outer_products: Matrix = np.einsum("ij,ik->ijk", A, A, optimize=True)
    rows_outer_products = rows_outer_products.reshape((n, -1)).T
    subrows, weights = fast_caratheodory_set_python(rows_outer_products, np.ones(n) / n, clusters_count)
    reduced_mat: Matrix = np.sqrt(np.multiply(subrows[0:d ** 2:d + 1, :], n * weights)).T
    return reduced_mat
