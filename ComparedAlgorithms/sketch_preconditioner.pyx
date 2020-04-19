# cython: language_level=3, boundscheck=False, wraparound=False
# cython: initializedcheck=False, cdivision=True, nonecheck=False
import numpy as np
cimport numpy as np
from libc.stdlib cimport rand, RAND_MAX, srand, malloc, free, abort
from cython.parallel cimport prange
from scipy.fft import dct
from Infrastructure.utils import RowVector, Matrix

cdef inline void _random_sign_change(const double[:, ::1] data_matrix, double[:, ::1] sketched_mat,
                                     const double switch_sign_probability) nogil:
    cdef Py_ssize_t rows_num = data_matrix.shape[0]
    cdef Py_ssize_t column_num = data_matrix.shape[1]
    cdef Py_ssize_t i, j

    for i in prange(rows_num):
        if <double>rand() / <double>RAND_MAX <= switch_sign_probability:
            for j in prange(column_num):
                sketched_mat[i, j] = -data_matrix[i, j]
        else:
            for j in prange(column_num):
                sketched_mat[i, j] = data_matrix[i, j]

cdef inline double[:, ::1] _pick_random_rows(double[:, ::1] sketched_mat, const unsigned int sampled_rows):
    cdef Py_ssize_t* picked_rows = <Py_ssize_t*> malloc(sizeof(Py_ssize_t) * sampled_rows)
    cdef Py_ssize_t[:] picked_rows_view
    cdef Py_ssize_t total_rows_num = sketched_mat.shape[0]
    cdef int row_index
    if picked_rows == NULL:
        abort()

    for row_index in prange(sampled_rows, nogil=True):
        picked_rows[row_index] = <Py_ssize_t>(rand() % total_rows_num)

    picked_rows_view = <Py_ssize_t[:sampled_rows]> picked_rows
    sketched_mat = sketched_mat.base[picked_rows_view, :]  # Unfortunately, this is a Python object call...
    free(picked_rows)
    return sketched_mat


def generate_sketch_preconditioner(const double[:, ::1] data_matrix, const unsigned int sampled_rows,
                                   double[:, ::1] sketched_mat, const unsigned int seed,
                                   const double switch_sign_probability):
    srand(seed)  # Seeding the random number generator.
    _random_sign_change(data_matrix, sketched_mat, switch_sign_probability)
    sketched_mat = dct(sketched_mat, norm='ortho')
    #return _pick_random_rows(sketched_mat, sampled_rows)  # TODO: Check run time in Google Colab.
    sketched_mat = np.array(sketched_mat)[np.random.randint(low=0, high=sketched_mat.shape[0], size=sampled_rows), :]
    return sketched_mat
