import unittest
import numpy as np


from mat import *

class Test_ProductMatrix(unittest.TestCase):
    def test_mult_zeros_ones(self):
        self.assertTrue(np.array_equal(product_matrix(np.zeros((5, 5)), np.ones((5, 5))), np.zeros((5, 5))))

    def test_mult_identity(self):
        identity_matrix = np.identity(3)
        self.assertTrue(np.array_equal(product_matrix(identity_matrix, identity_matrix), identity_matrix))

    def test_mult_random(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[2, 0], [1, 2]])
        expected_result = np.array([[4, 4], [10, 8]])
        self.assertTrue(np.array_equal(product_matrix(A, B), expected_result))

    

    def test_mult_large_matrices(self):
        A = np.random.rand(100, 100)
        B = np.random.rand(100, 100)
        expected_result = np.dot(A, B)
        self.assertTrue(np.allclose(product_matrix(A, B), expected_result))

    def test_mult_non_square_matrices(self):
        A = np.random.rand(50, 100)
        B = np.random.rand(100, 50)
        expected_result = np.dot(A, B)
        self.assertTrue(np.allclose(product_matrix(A, B), expected_result))

    def test_mult_large_identity(self):
        identity_matrix = np.identity(100)
        self.assertTrue(np.array_equal(product_matrix(identity_matrix, identity_matrix), identity_matrix))

    def test_mult_large_zeros_ones(self):
        self.assertTrue(np.array_equal(product_matrix(np.zeros((100, 100)), np.ones((100, 100))), np.zeros((100, 100))))

class Test_Det(unittest.TestCase):
    def test_det_2x2(self):
        matrix = np.array([[4, 6], [3, 8]])
        self.assertEqual(det(matrix), 14)

    def test_det_3x3(self):
        matrix = np.array([[6, 1, 1], [4, -2, 5], [2, 8, 7]])
        self.assertEqual(det(matrix), -306)

    def test_det_identity(self):
        identity_matrix = np.identity(4)
        self.assertEqual(det(identity_matrix), 1)

    def test_det_zero_matrix(self):
        zero_matrix = np.zeros((3, 3))
        self.assertEqual(det(zero_matrix), 0)

    def test_det_singular_matrix(self):
        singular_matrix = np.array([[2, 4], [1, 2]])
        self.assertEqual(det(singular_matrix), 0)

    def test_det_large_matrix(self):
        large_matrix = np.random.rand(10, 10)
        expected_result = np.linalg.det(large_matrix)
        self.assertAlmostEqual(det(large_matrix), expected_result, places=5)
    def test_det_wrong_shape(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            det(matrix)


class Test_UpBackSolution(unittest.TestCase):
    def test_up_back_solution_2x2(self):
        matrix = np.array([[2, 3], [0, 1]])
        coefficients = np.array([5, 1])
        expected_result = np.array([1, 1])
        self.assertTrue(np.allclose(up_back_solution(matrix, coefficients), expected_result))

    def test_up_back_solution_3x3(self):
        matrix = np.array([[3, 2, -1], [0, 1, 2], [0, 0, 1]])
        coefficients = np.array([1, 2, 3])
        expected_result = np.array([4, -4, 3])
        self.assertTrue(np.allclose(up_back_solution(matrix, coefficients), expected_result))

    def test_up_back_solution_identity(self):
        identity_matrix = np.identity(4)
        coefficients = np.array([1, 2, 3, 4])
        expected_result = coefficients
        self.assertTrue(np.allclose(up_back_solution(identity_matrix, coefficients), expected_result))

    def test_up_back_solution_large_matrix(self):
        matrix = np.triu(np.random.rand(100, 100))
        coefficients = np.random.rand(100)
        expected_result = np.linalg.solve(matrix, coefficients)
        self.assertTrue(np.allclose(up_back_solution(matrix, coefficients), expected_result))

    def test_up_back_solution_singular_matrix(self):
        matrix = np.array([[1, 2], [0, 0]])
        coefficients = np.array([1, 0])
        with self.assertRaises(np.linalg.LinAlgError):
            up_back_solution(matrix, coefficients)

    def test_up_back_solution_wrong_shape(self):
        matrix = np.array([[1, 2, 3], [0, 1, 2]])
        coefficients = np.array([1, 2])
        with self.assertRaises(TypeError):
            up_back_solution(matrix, coefficients)

if __name__ == '__main__':
    unittest.main()