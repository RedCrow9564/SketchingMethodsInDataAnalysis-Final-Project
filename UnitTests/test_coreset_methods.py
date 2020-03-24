"""
test_coreset_methods.py - Tests for coreset computation methods
===============================================================

This module contains the tests for coreset computation methods.
"""
import unittest
import numpy as np
from Infrastructure.utils import Matrix
from ComparedAlgorithms.method_boosters import create_coreset_fast_caratheodory


class TestMatrixCoreset(unittest.TestCase):
    """ A class which contains tests for methods for coreset computation."""

    def test_outer_products_decomposition(self):
        """
        This test verifies that this `einsum` function computes the outer products of all the matrix's rows.
        This means the sum of these outer products must be equal to the original matrix.
        """
        n: int = 10000
        d: int = 15
        A: Matrix = np.random.rand(n, d)
        rows_outer_products: Matrix = np.einsum("ij,ik->ijk", A, A, optimize=True)
        self.assertTrue(np.allclose(A.T.dot(A), rows_outer_products.sum(axis=0), atol=1e-10, rtol=0))

    def test_python_fast_matrix_coresets(self):
        """
        This test verifies the 'coreset' of the matrix has the same Gram matrix as the original matrix, i.e
        :math:`L^{T}L=A^{T}A`.
        """
        n: int = 2000
        d: int = 3
        clusters_count: int = 20
        A: Matrix = np.random.rand(n, d)
        reduced_mat: Matrix = create_coreset_fast_caratheodory(A, clusters_count)
        self.assertTrue(np.allclose(reduced_mat.T.dot(reduced_mat), A.T.dot(A), atol=1e-10, rtol=0))


if __name__ == '__main__':
    unittest.main()
