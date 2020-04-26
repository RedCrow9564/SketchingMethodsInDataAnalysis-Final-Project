"""
test_caratheodory_set.py - Tests for Caratheodory subset computation methods
============================================================================

This module contains the tests for the methods which are responsible for Caratheodory subset computation.
"""
import unittest
import numpy as np
from Infrastructure.utils import Scalar, Vector, Matrix
from ComparedAlgorithms.method_boosters import fast_caratheodory_set_python
from ComparedAlgorithms.basic_caratheodory_set import caratheodory_set_python


class TestCaratheodorySet(unittest.TestCase):
    """ A class which contains tests for Caratheodory subset computation."""

    def test_python_Caratheodory_set(self):
        """
        This method tests the basic algorithm for Caratheodory subset algorithm.
        """
        d: int = 15
        n: int = 100
        data: Matrix = np.random.rand(d, n)
        weights: Vector = np.random.random(n)
        weights /= weights.sum()
        epsilon: Scalar = np.finfo(np.float).eps
        new_weights, chosen_indices = caratheodory_set_python(data, weights)
        self.assertAlmostEqual(np.sum(weights), 1, delta=5 * epsilon)
        self.assertTrue(np.all(weights >= -epsilon), f'\nSome negative weights: {weights[np.invert(weights >= 0)]}')
        self.assertLessEqual(len(chosen_indices), d + 1)
        self.assertTrue(np.allclose(data.dot(weights), data[:, chosen_indices].dot(new_weights), atol=1e-10, rtol=0))

    def test_fast_python_Caratheodory_set(self):
        """
        This method tests the faster algorithm for Caratheodory subset algorithm.
        """
        d: int = 3
        n: int = 50
        clusters_count: int = 2 * (d + 1) ** 2 + 2
        data: Matrix = np.asfortranarray(np.random.rand(d, n))
        weights: Vector = np.random.random(n)
        weights /= weights.sum()
        epsilon: Scalar = np.finfo(np.float).eps
        new_weights, chosen_indices = fast_caratheodory_set_python(data, weights, clusters_count)
        self.assertAlmostEqual(np.sum(weights), 1, delta=5 * epsilon)
        self.assertTrue(np.all(weights >= -epsilon), f'\nSome negative weights: {weights[np.invert(weights >= 0)]}')
        self.assertLessEqual(len(chosen_indices), d + 1)
        self.assertTrue(np.allclose(data.dot(weights), data[:, chosen_indices].dot(new_weights), atol=1e-10, rtol=0))


if __name__ == '__main__':
    unittest.main()
