import unittest
import numpy as np
from Infrastructure.utils import Scalar, Vector, ColumnVector, Matrix
from ComparedAlgorithms.method_boosters import _caratheodory_set_python, fast_caratheodory_set_python


class TestCaratheodorySet(unittest.TestCase):

    def test_python_Caratheodory_set(self):
        d: int = 15
        n: int = 100
        data: Matrix = np.random.rand(d, n)
        weights: Vector = np.random.random(n)
        weights /= weights.sum()
        epsilon: Scalar = np.finfo(np.float).eps
        subset, new_weights = _caratheodory_set_python(data, weights)
        self.assertAlmostEqual(np.sum(weights), 1, delta=5 * epsilon)
        self.assertTrue(np.all(weights >= -epsilon),
                        f'\nSome negative weights: {weights[np.invert(weights >= 0)]}')
        self.assertLessEqual(len(subset), d + 1)
        self.assertTrue(np.allclose(data.dot(weights), subset.dot(new_weights),
                                    atol=1e-10, rtol=0))
        

    def test_fast_python_Caratheodory_set(self):
        d: int = 15
        n: int = 10000
        clusters_count: int = 20
        data: Matrix = np.random.rand(d, n)
        weights: Vector = np.random.random(n)
        weights /= weights.sum()
        epsilon: Scalar = np.finfo(np.float).eps
        subset, new_weights = fast_caratheodory_set_python(data, weights, clusters_count)
        self.assertAlmostEqual(np.sum(weights), 1, delta=5 * epsilon)
        self.assertTrue(np.all(weights >= -epsilon),
                        f'\nSome negative weights: {weights[np.invert(weights >= 0)]}')
        self.assertLessEqual(len(subset), d + 1)
        self.assertTrue(np.allclose(data.dot(weights), subset.dot(new_weights),
                                    atol=1e-10, rtol=0))


if __name__ == '__main__':
    unittest.main()
