import unittest
import numpy as np
from Infrastructure.utils import Scalar, Vector, ColumnVector, Matrix
from ComparedAlgorithms.method_boosters import create_coreset_fast_caratheodory


class TestMatrixCoreset(unittest.TestCase):

    def test_outer_products_decomposition(self):
        n: int = 10000
        d: int = 15
        A: Matrix = np.random.rand(n, d)
        rows_outer_products: Matrix = np.einsum("ij,ik->ijk", A, A, optimize=True)
        self.assertTrue(np.allclose(
            A.T.dot(A), rows_outer_products.sum(axis=0), atol=1e-10, rtol=0))

    def test_python_fast_matrix_coresets(self):
        n: int = 2000
        d: int = 3
        clusters_count: int = 20
        failes_counter: int = 0
        solution_found: bool = False
        while not solution_found:
            A: Matrix = np.random.rand(n, d)
            try:
                reduced_mat: Matrix = create_coreset_fast_caratheodory(A, clusters_count)
                solution_found = True
            except np.linalg.LinAlgError as e:
                if 'Singular matrix' in str(e):
                    failes_counter += 1
                    continue
                else:
                    raise e
        print(f'\nFailed reducing attempts: {failes_counter}')
        self.assertTrue(np.allclose(
            reduced_mat.T.dot(reduced_mat), A.T.dot(A), atol=1e-10, rtol=0))


if __name__ == '__main__':
    unittest.main()
