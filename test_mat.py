import unittest
import numpy as np
import scipy.linalg as la


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

class Test_DownBackSolution(unittest.TestCase):
    def test_down_back_solution_2x2(self):
        matrix = np.array([[2, 0], [3, 1]])
        coefficients = np.array([4, 7])
        expected_result = np.array([2, 1])
        self.assertTrue(np.allclose(down_back_solution(matrix, coefficients), expected_result))

    def test_down_back_solution_3x3(self):
        matrix = np.array([[1, 0, 0], [2, 1, 0], [3, 4, 1]])
        coefficients = np.array([1, 4, 10])
        expected_result = np.array([1, 2, -1])
        self.assertTrue(np.allclose(down_back_solution(matrix, coefficients), expected_result))

    def test_down_back_solution_identity(self):
        identity_matrix = np.identity(4)
        coefficients = np.array([1, 2, 3, 4])
        expected_result = coefficients
        self.assertTrue(np.allclose(down_back_solution(identity_matrix, coefficients), expected_result))

    def test_down_back_solution_large_matrix(self):
        matrix = np.tril(np.random.rand(10,10))
        coefficients = np.random.rand(10)
        expected_result = np.linalg.solve(matrix, coefficients)
        print(down_back_solution(matrix, coefficients) - expected_result)
        self.assertTrue(np.allclose(down_back_solution(matrix, coefficients), expected_result))

    def test_down_back_solution_singular_matrix(self):
        matrix = np.array([[1, 0], [0, 0]])
        coefficients = np.array([1, 0])
        with self.assertRaises(np.linalg.LinAlgError):
            down_back_solution(matrix, coefficients)

    def test_down_back_solution_wrong_shape(self):
        matrix = np.array([[1, 2, 3], [0, 1, 2]])
        coefficients = np.array([1, 2])
        with self.assertRaises(TypeError):
            down_back_solution(matrix, coefficients)
class Test_Gauss(unittest.TestCase):
    def test_gauss_2x2(self):
        matrix = np.array([[2, 1], [1, 3]])
        coefficients = np.array([4, 5])
        expected_matrix = np.array([[2, 1], [0, 2.5]])
        expected_coefficients = np.array([4, 3])
        result_matrix, result_coefficients, _ = gauss(matrix, coefficients, partial_pivoting=False)
        self.assertTrue(np.allclose(result_matrix, expected_matrix))
        self.assertTrue(np.allclose(result_coefficients, expected_coefficients))

    def test_gauss_3x3(self):
        matrix = np.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]])
        coefficients = np.array([4, 5, 6])
        expected_matrix = la.lu(matrix , True)[1]
        expected_coefficients = np.array([4, 3, 4.6])
        result_matrix, result_coefficients, _ = gauss(matrix, coefficients, partial_pivoting=False)
        self.assertTrue(np.allclose(result_matrix, expected_matrix))
        self.assertTrue(np.allclose(result_coefficients, expected_coefficients))

    def test_gauss_partial_pivoting(self):
        matrix = np.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]])
        coefficients = np.array([1, 0, 1])
        expected_matrix = la.lu(matrix , False)[1]
        
        result_matrix, result_coefficients, _ = gauss(matrix, coefficients, partial_pivoting=False)
        self.assertTrue(np.allclose(result_matrix, expected_matrix))
        

    def test_gauss_large_matrix(self):
        matrix = np.random.rand(100, 100)
        coefficients = np.random.rand(100)
        result_matrix, result_coefficients, _ = gauss(matrix, coefficients, partial_pivoting=True)
        expected_matrix, expected_coefficients = np.triu(result_matrix), result_coefficients
        self.assertTrue(np.allclose(result_matrix, expected_matrix))
        self.assertTrue(np.allclose(result_coefficients, expected_coefficients))

    def test_gauss_wrong_shape(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        coefficients = np.array([1, 2])
        with self.assertRaises(TypeError):
            gauss(matrix, coefficients)


class Test_SolveMat(unittest.TestCase):
    def test_solve_mat_2x2(self):
        matrix = np.array([[2, 1], [1, 3]])
        coefficients = np.array([4, 5])
        expected_result = np.linalg.solve(matrix, coefficients)
        self.assertTrue(np.allclose(solve_mat(matrix, coefficients), expected_result))

    def test_solve_mat_3x3(self):
        matrix = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
        coefficients = np.array([1, 0, 1])
        expected_result = np.linalg.solve(matrix, coefficients)
        self.assertTrue(np.allclose(solve_mat(matrix, coefficients), expected_result))

    def test_solve_mat_identity(self):
        identity_matrix = np.identity(4)
        coefficients = np.array([1, 2, 3, 4])
        expected_result = np.linalg.solve(identity_matrix, coefficients)
        self.assertTrue(np.allclose(solve_mat(identity_matrix, coefficients), expected_result))

    def test_solve_mat_large_matrix(self):
        matrix = np.random.rand(100, 100)
        coefficients = np.random.rand(100)
        expected_result = np.linalg.solve(matrix, coefficients)
        self.assertTrue(np.allclose(solve_mat(matrix, coefficients), expected_result))

    def test_solve_mat_singular_matrix(self):
        matrix = np.array([[1, 2], [2, 4]])
        coefficients = np.array([1, 2])
        with self.assertRaises(np.linalg.LinAlgError):
            solve_mat(matrix, coefficients)

    def test_solve_mat_wrong_shape(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        coefficients = np.array([1, 2])
        with self.assertRaises(TypeError):
            solve_mat(matrix, coefficients)

            
class Test_InverseMat(unittest.TestCase):
    def test_inverse_mat_2x2(self):
        matrix = np.array([[4, 7], [2, 6]])
        expected_result = np.linalg.inv(matrix)
        self.assertTrue(np.allclose(inverse_mat(matrix), expected_result))

    def test_inverse_mat_3x3(self):
        matrix = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
        expected_result = np.linalg.inv(matrix)
        self.assertTrue(np.allclose(inverse_mat(matrix), expected_result))

    def test_inverse_mat_identity(self):
        identity_matrix = np.identity(4)
        expected_result = np.linalg.inv(identity_matrix)
        self.assertTrue(np.allclose(inverse_mat(identity_matrix), expected_result))

    def test_inverse_mat_large_matrix(self):
        matrix = np.random.rand(10, 10)
        expected_result = np.linalg.inv(matrix)
        self.assertTrue(np.allclose(inverse_mat(matrix), expected_result))

    def test_inverse_mat_singular_matrix(self):
        matrix = np.array([[1, 2], [2, 4]])
        with self.assertRaises(np.linalg.LinAlgError):
            inverse_mat(matrix)

    def test_inverse_mat_wrong_shape(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(TypeError):
            inverse_mat(matrix)


class Test_LUDecomposition(unittest.TestCase):
    def test_lu_decomposition_2x2(self):
        matrix = np.array([[4, 3], [6, 3]])
        P, L, U = LU_decomposition(matrix)
        expected_P, expected_L, expected_U = la.lu(matrix)
        self.assertTrue(np.allclose(P@L@U , expected_P@expected_L@expected_U))

    def test_lu_decomposition_3x3(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        P, L, U = LU_decomposition(matrix)
        expected_P, expected_L, expected_U = la.lu(matrix)
        self.assertTrue(np.allclose(P@L@U , expected_P@expected_L@expected_U))

    def test_lu_decomposition_identity(self):
        identity_matrix = np.identity(4)
        P, L, U = LU_decomposition(identity_matrix)
        self.assertTrue(np.allclose(L, identity_matrix))
        self.assertTrue(np.allclose(U, identity_matrix))
        self.assertTrue(np.allclose(P @ identity_matrix, L @ U))

    def test_lu_decomposition_large_matrix(self):
        matrix = np.random.rand(10, 10)
        P, L, U = LU_decomposition(matrix)
        expected_P, expected_L, expected_U = la.lu(matrix)
        self.assertTrue(np.allclose(P @ L @ U, expected_P @ expected_L @ expected_U))


    def test_lu_decomposition_wrong_shape(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(TypeError):
            LU_decomposition(matrix)


class Test_LUDeterminant(unittest.TestCase):
    def test_lu_determinant_2x2(self):
        matrix = np.array([[4, 7], [2, 6]])
        expected_result = np.linalg.det(matrix)
        self.assertAlmostEqual(LU_determinant(matrix), expected_result, places=5)

    def test_lu_determinant_3x3(self):
        matrix = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
        expected_result = np.linalg.det(matrix)
        self.assertAlmostEqual(LU_determinant(matrix), expected_result, places=5)

    def test_lu_determinant_identity(self):
        identity_matrix = np.identity(4)
        expected_result = np.linalg.det(identity_matrix)
        self.assertAlmostEqual(LU_determinant(identity_matrix), expected_result, places=5)

    def test_lu_determinant_large_matrix(self):
        matrix = np.random.rand(10, 10)
        expected_result = np.linalg.det(matrix)
        self.assertAlmostEqual(LU_determinant(matrix), expected_result, places=5)

    def test_lu_determinant_singular_matrix(self):
        matrix = np.array([[1, 2], [2, 4]])
        with self.assertRaises(np.linalg.LinAlgError):
            LU_determinant(matrix)

    def test_lu_determinant_wrong_shape(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(TypeError):
            LU_determinant(matrix)

    
if __name__ == '__main__':
    unittest.main()