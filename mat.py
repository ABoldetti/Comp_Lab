import numpy as np

#
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

    x = []
    for index_x in range(len(matrix)):
        sum = 0
        for index_sum in range(index_x):
            sum+= matrix[index_x , index_sum]*x[index_sum]
        x.append( (coefficents[index_x]-sum)/matrix[index_x,index_x])
    return np.array(x)

def gauss( matrix , b , partial_pivoting:bool = True):
    """
    Perform Gaussian elimination on the matrix. Modify the variable globally

    Parameters
    ----------
    mat : 2D array-like
        The input matrix.
    b : 1D array-like
        The known value array.
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
    if len(matrix) != len(b): raise TypeError("the 2 argument has to be the same lenght")
    if len(matrix) != len(matrix[0]): raise TypeError("the matrix has to be square")

    if(partial_pivoting):part_pivot( matrix , b )

    for i in range( len(matrix)-1):
        for j in range( i+1 , len(matrix)):
            k = matrix[i,i] / matrix[j,i]
            matrix[j] = k * matrix[j] - matrix[i]
            b[j] = k * b[j] - b[i]

def part_pivot(matrix , b):
    """
    Perform partial pivoting on the matrix. Modify globally the matrix

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
    """
    if len(matrix) != len(b): raise TypeError("the 2 argument has to be the same lenght")
    if len(matrix) != len(matrix[0]): raise TypeError("the matrix has to be square")

    for i in range( len(matrix)-1):
        for j in range( i+1 , len(matrix)-1):
            if matrix[j,i] > matrix[i,i]:
                matrix[i],matrix[j] = matrix[j],matrix[i]
                b[i],b[j] = b[j],b[i]

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
            k = mat1[i,i] / mat1[j,i]
            mat1[j] = k * mat1[j] - mat1[i]
            inv_mat[j] = k * inv_mat[j] - inv_mat[i]
    inv_mat = np.transpose(inv_mat)
    mat1 = np.transpose( mat1)
    
    for i in range( len(mat1)-1):
        for j in range( i+1 , len(mat1)):
            k = mat1[i,i] / mat1[j,i]
            mat1[j] = k * mat1[j] - mat1[i]
            inv_mat[j] = k * inv_mat[j] - inv_mat[i]

    for i in range( len(mat1) ):
        inv_mat[i] /= mat1[i,i]


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


if __name__ == '__main__':
    mat = np.array([[2,1,5],[1,8,-2],[1,2,1]] , dtype=np.float32)
    b = np.array( [1,-2,2] , dtype = np.float32)

    mat.dtype = np.float32
    #rnd_mat = 5*np.random.rand( 3,3)-2.5
    #rnd_b = 10*np.random.rand(3) - 5
    
    print(solve_mat(mat,b))
    print(inverse_mat(mat))
    print(mat)

    print( np.matmul(inverse_mat(mat) , mat))