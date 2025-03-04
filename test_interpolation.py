import unittest
import numpy as np
from Comp_Lab.interpolation import interpolation, vandermonde_matrix, poly

class TestInterpolation(unittest.TestCase):
    def test_interpolation_linear(self):
        x = np.array([1, 2, 3, 4])
        y = np.array([1, 2, 3, 4])
        coefficients = interpolation(x, y)
        expected_coefficients = np.array([0, 1 , 0, 0])
        self.assertTrue(np.allclose(coefficients, expected_coefficients , atol = 1e-6))

    def test_interpolation_quadratic(self):
        x = np.array([1, 2, 3, 4])
        y = np.array([1, 4, 9, 16])
        coefficients = interpolation(x, y)
        expected_coefficients = np.array([0, 0, 1 , 0])
        self.assertTrue(np.allclose(coefficients, expected_coefficients , atol = 1e-6))

    def test_interpolation_vandermonde_matrix(self):
        x = np.array([1, 2, 3])
        y = np.array([1, 4, 9])
        v_matrix = vandermonde_matrix(x, y)
        expected_matrix = np.array([[1, 1, 1], [1, 2, 4], [1, 3, 9]])
        self.assertTrue(np.allclose(v_matrix, expected_matrix))

    def test_interpolation_poly(self):
        x = np.array([1, 2, 3, 4])
        y = np.array([1, 4, 9, 16])
        coefficients = interpolation(x, y)
        x_values = np.linspace(1, 4, 100)
        y_values = poly(x_values, coefficients)
        expected_y_values = x_values**2
        self.assertTrue(np.allclose(y_values, expected_y_values))

    def test_interpolation_different_lengths(self):
        x = np.array([1, 2, 3])
        y = np.array([1, 4])
        with self.assertRaises(ValueError):
            interpolation(x, y)

if __name__ == '__main__':
    unittest.main()