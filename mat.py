import numpy as np
import scipy as sp

def product_matrix( A , B ):
    """
    Calculate the product row by column between two matrix

    Parameters
    ----------
    A: 2D array like
        first matrix
    B: 2D array like
        second matrix
    
    Raises
    ------
    1: len(A[0]) != len(B)
            First matrix has to have the same row lenght as the second matrix's column lenght

    Returns
    -------
    np.array
        the resulting matrix
    """
    if len(A[0]) != len(B):
        raise ValueError("First matrix has to have the same row lenght as the second matrix's column lenght")
    
    prod_mat = np.zeros( (len(A) , len(B[0])))
    
    for new_mat_col in range(len(A)):
        for new_mat_row in range(len(B[0])):
            for j in range(len(B)):
                prod_mat[new_mat_row , new_mat_col] += B[j,new_mat_col] * A[new_mat_row,j]
    return prod_mat


def det(matrix):
    """
    Calculate the determinant of a square matrix.

    Parameters
    ----------
    mat : 2D array-like
        The input matrix.
    
    Raises
    ------
    1: len(mat) == len(mat[0])
            Matrix has to be square

    Returns
    -------
    float
        The determinant of the matrix.
    """
    # Check if the matrix is square
    if len(matrix) != len(matrix[0]):
        raise ValueError("Matrix must be square")

    # Base case for 2x2 matrix
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Recursive case for larger matrices
    determinant = 0
    for c in range(len(matrix)):
        sub_mat = np.delete(np.delete(matrix, 0, axis=0), c, axis=1)
        determinant += ((-1) ** c) * matrix[0][c] * det(sub_mat)

    return determinant

def up_back_solution( matrix:np.array , coefficents:np.array ) -> np.array:
    """
    Solve a matrix with backward substitution of UpperTriangular Matrix.

    Parameters
    ----------
    mat : 2D array-like
        The input matrix.
    b : 1D array-like
        The known value array.
    
    Raises
    ------
    1: len(mat) == len(b)
            matrix and vector has to have the same lenght
    2: len(mat) == len(mat[0])
            Matrix has to be square

    Returns
    -------
    np.array
        The solution vector.
    """
    if len(matrix) != len(coefficents): raise TypeError("the 2 argument has to be the same lenght")
    if len(matrix) != len(matrix[0]): raise TypeError("the matrix has to be square")
    for i in matrix:
        if np.array_equal(i ,np.zeros(len(matrix))): raise np.linalg.LinAlgError("Matrix is singular")

    x = []
    for index_x in range(len(matrix)):
        sum = 0
        for index_sum in range(index_x):
            a = 0
            sum+= matrix[len(matrix) - 1 - index_x , len(matrix) - 1 -index_sum]*x[index_sum]
        x.append( (coefficents[len(matrix) - 1 - index_x]-sum)/matrix[len(matrix) - 1 - index_x , len(matrix) - 1 - index_x])
    return np.flip(x)

def down_back_solution( matrix, coefficents ):
    """
    Solve a matrix with forward substitution of LowerTriangular Matrix.

    Parameters
    ----------
    mat : 2D array-like
        The input matrix.
    b : 1D array-like
        The known value array.

    Raises
    ------
    1: len(mat) == len(b)
            matrix and vector has to have the same lenght
    2: len(mat) == len(mat[0])
            Matrix has to be square
    Returns
    -------
    np.array
        The solution vector.
    """
    if len(matrix) != len(coefficents): raise TypeError("the 2 argument has to be the same lenght")
    if len(matrix) != len(matrix[0]): raise TypeError("the matrix has to be square")
    for i in matrix:
        if np.array_equal(i ,np.zeros(len(matrix))): raise np.linalg.LinAlgError("Matrix is singular")

    x = []
    for index_x in range(len(matrix)):
        sum = 0
        for index_sum in range(index_x):
            sum+= matrix[index_x , index_sum]*x[index_sum]
        x.append( (coefficents[index_x]-sum)/matrix[index_x,index_x])
    return np.array(x)

def gauss( matrix , b = 0, partial_pivoting:bool = True , permutation:bool = False):
    """
    Perform Gaussian elimination on the matrix till it gets an upper triangular matrix.
    Modify the variable globally

    Parameters
    ----------
    matrix : 2D array-like
        The input matrix.
    b : 1D array-like 
        The known value array.  
        (Optional) Default: zero filled array of the size of mat
    partial_pivoting: bool
        if you want partial pivoting or not.    
        Default: True
    
    Raises
    ------
    1: len(mat) == len(b)
            matrix and vector has to have the same lenght
    2: len(mat) == len(mat[0])
            Matrix has to be square

    """
    if (type(b) is int) and (b == 0): b = np.zeros(len(matrix))

    if len(matrix) != len(b): raise TypeError("the 2 argument has to be the same lenght")
    if len(matrix) != len(matrix[0]): raise TypeError("the matrix has to be square")
    

    if(partial_pivoting): p = part_pivot( matrix , b , permutation)

    for i in range( len(matrix)-1):
        for j in range( i+1 , len(matrix)):

            k = matrix[j,i] / matrix[i,i]
            matrix[j] = -k * matrix[i] + matrix[j]
            b[j] = -k * b[j] + b[i]
    return p


def part_pivot(matrix , b = 0 , permutation = False):

    """
    Perform partial pivoting on the matrix. Modify globally the matrix

    Parameters
    ----------
    mat : 2D array-like
        The input matrix.
    b : 1D array-like
        The known value array.
        Default: 0, it becomes an array full of zeros of length = len(matrix)
    
    Raises
    ------
    1: len(mat) == len(b)
            matrix and vector has to have the same lenght
    2: len(mat) == len(mat[0])
            Matrix has to be square
    """
    if (type(b) is int) and (b == 0): b = np.zeros(len(matrix))
    
    if len(matrix) != len(b): raise TypeError("the 2 argument has to be the same lenght")
    if len(matrix) != len(matrix[0]): raise TypeError("the matrix has to be square")

    I = np.identity(len(matrix))
    for i in range( len(matrix)):
        for j in range( i+1 , len(matrix)):
            if matrix[j,i] > matrix[i,i]:
                matrix[[i,j]] = matrix[[j,i]]
                I[[i,j]] = I[[j,i]]
                b[[i,j]] = b[[j,i]]
    if (permutation): return I

def inverse_mat( matrix:np.array) -> np.array:
    """
    Function that takes a square matrix and output the inverse mat
    
    Parameters
    -----------
    mat: 2D array like
        matrix to invert
    
    Returns
    --------
    output: 2D array like
        inverse matrix
        
    Rises
    --------
    1: len(mat) == len(mat[0])
            Matrix has to be square
    """



    if len(matrix) != len(matrix[0]): raise TypeError("the matrix has to be square")

    mat1 = np.copy(matrix)
    inv_mat = np.identity(len(mat1) , dtype= np.float32)
    for i in range( len(mat1)-1):
        for j in range( i+1 , len(mat1)):
            k = mat1[j,i] / mat1[i,i]
            mat1[j] = -k * mat1[i] + mat1[j]
            inv_mat[j] = -k * inv_mat[i] + inv_mat[j]
    inv_mat = np.transpose(inv_mat)
    mat1 = np.transpose( mat1)
    
    for i in range( len(mat1)-1):
        for j in range( i+1 , len(mat1)):
            k = mat1[j,i] / mat1[i,i]
            mat1[j] = -k * mat1[i] + mat1[j]
            inv_mat[j] = -k * inv_mat[i] + inv_mat[j]

    for i in range( len(mat1) ):
        inv_mat[i] /= mat1[i,i]


    return np.transpose(inv_mat)


def inverse( matrix: np.array) -> np.array:
    
    if len(matrix) != len(matrix[0]): raise TypeError("the matrix has to be square")

    mat1 = np.copy(matrix)
    id = np.identity(len(mat1) , dtype= np.float32)
    inv_mat = np.copy(matrix)
    
    for i in range(len(matrix)):
        inv_mat[i] = solve_mat( matrix , id[i])
    return np.transpose(inv_mat)


def solve_mat( matrix:np.array , coefficents:np.array , change_mat:bool = False , partial_pivoting:bool = True) -> np.array:
    """
    Parameters
    ---------
    matrix: 2D array like
        matrix of coefficients, M in the formula Mx = b
    coefficients: 1D array like
        list of coefficients, b in the formula Mx = b
    change_mat: bool
        bool value if you want the input mat and b changed or not.
        Default: False
    partial_pivoting: bool
        bool value if you want to apply partial pivoting to the gauss method.
        Default:True
    Returns
    ---------
    x: 1D array like
        list of solution for the matrix, x in the formula Mx = b

    Raises
    ------
    1: len(mat) == len(b)
            matrix and vector has to have the same lenght
    2: len(mat) == len(mat[0])
            Matrix has to be square
    
    """
    
    if change_mat:
        mat1 = matrix
        b = coefficents
    else:
        mat1 = np.copy(matrix)
        b = np.copy(coefficents)
    
    gauss( mat1 , b , partial_pivoting)
    return up_back_solution( mat1 , b)


def LU_decomposition( mat:np.array , partial_pivot:bool = True) -> list:
    """
    Function that takes a square matrix and write it as A = LU
    
    Parameters
    -----------
    mat: 2D array like
        matrix to decompose
    partial_pivot: bool
        Bool value, if true acts with partial pivoting on the matrix
        Default: True
    
    Returns
    --------
    P: 2D array like
        permutation matrix
        if partial pivot = False: Identity
    L: 2D array like
        Lower triangular matrix, first in the product
    U: 2D array like
        Upper triangular matrix, second in the product
        
    Rises
    --------
    1: len(mat) == len(mat[0])
            Matrix has to be square
    """

    if len(mat) != len(mat[0]) : raise TypeError("matrix has to be square")
    
    L = np.identity(len(mat))
    U = np.copy(mat)

    P = part_pivot(U , permutation=partial_pivot)

    for i in range( len(U)-1):
        for j in range( i+1 , len(U)):

            L[j,i]= U[j,i] / U[i,i]
            U[j] = -L[j,i] * U[i] + U[j]
    
    if type(P) == type(None): P = np.identity(len(mat))
    return [P,L,U]

def LU_determinant( matrix: np.array) -> float:
    """
    Function that takes a square matrix and calculate the determinant using LU decomposition
    
    Parameters
    -----------
    mat: 2D array like
        matrix to decompose
    
    Returns
    --------
    output: float
    determinant
    Rises
    --------
    1: len(mat) == len(mat[0])
            Matrix has to be square
    """
    if len(matrix) != len(matrix[0]) : raise TypeError("matrix has to be square")

    P,_,U = LU_decomposition(matrix)
    return det(P)*np.prod( [U[i,i] for i in range(len(U))])



if __name__ == '__main__':
    matrix = np.array([[1, 2], [0, 0]])
    coefficients = np.array([1, 0])
    print(up_back_solution(matrix, coefficients))    