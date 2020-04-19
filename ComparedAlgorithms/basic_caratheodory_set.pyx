# distutils: extra_compile_args = /fopenmp
# distutils: extra_link_args = /fopenmp
# cython: language_level=3, boundscheck=False, wraparound=False
# cython: initializedcheck=False, cdivision=True, nonecheck=False
cimport openmp
import numpy as np
cimport numpy as np
from cython.parallel cimport prange
from numpy.math cimport INFINITY


def cluster_average(const double[::1, :] cluster, const double[::1] weights):
    cdef Py_ssize_t cluster_size = cluster.shape[1], d = cluster.shape[0], i, j
    cdef double[::1] cluster_mean = np.empty(d)
    cdef double weights_sum = get_vector_sum(weights)

    for i in prange(d, nogil=True, schedule="static"):
        cluster_mean[i] = 0
        for j in range(cluster_size):
            cluster_mean[i] += weights[j] * cluster[i, j]
        cluster_mean[i] /= weights_sum

    return cluster_mean, weights_sum


cdef inline double get_vector_sum(const double[::1] vector) nogil:
    cdef double vector_sum = 0.0
    cdef Py_ssize_t i, n = vector.shape[0]

    for i in prange(n, schedule="static"):
        vector_sum += vector[i]

    return vector_sum

def caratheodory_set_python(np.ndarray[double, ndim=2, mode='c'] points,
                            np.ndarray[double, ndim=1, mode='c'] weights):
    """
    This method computes Caratheodory subset for the columns of a :math:`dxn` Matrix, with given weights.

    Args:
        points(Matrix): A :math:`dxn` Matrix.
        weights(ColumnVector): A weights for the columns of :math:`A`.

    Returns:
        A Caratheodory subset of :math:`d^{2]+1` columns of :math:`A, as a :math:`dx(d^{2}+1)` Matrix,
        and their weights as a ColumnVector.
    """
    cdef Py_ssize_t dim = len(points)  # The dimension of all rows, or rows of the matrix, d.
    cdef Py_ssize_t left_args = points.shape[1], i, j, n = points.shape[1], start_index
    cdef double alpha, temp_div, v_sum
    cdef np.ndarray[double, ndim=2] diff_points
    cdef np.ndarray[double, ndim=1] v
    cdef np.ndarray left_indices = np.arange(0, n, dtype=int)

    while left_args > dim + 1:
        diff_points = points[:, left_indices]
        diff_points = (diff_points[:, 1:].T - diff_points[:, 0].T).T

        v = np.linalg.svd(diff_points, full_matrices=True)[2][left_args - 2]

        v_sum = get_vector_sum(v)
        v = np.insert(v, [0], -v_sum)

        alpha = INFINITY
        for i in range(left_args):
            if v[i] > 0:
                temp_div = weights[left_indices[i]] / v[i]
                if temp_div < alpha:
                    alpha = temp_div

        weights[left_indices] -= alpha * v
        left_indices = np.nonzero(weights > 1e-15)[0]
        left_args = len(left_indices)

    return np.array(weights)[left_indices], left_indices
