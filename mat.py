import numpy as np

def det(mat):
    # Check if the matrix is square
    if len(mat) != len(mat[0]):
        raise ValueError("Matrix must be square")

    # Base case for 2x2 matrix
    if len(mat) == 2:
        return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]

    # Recursive case for larger matrices
    determinant = 0
    for c in range(len(mat)):
        sub_mat = np.delete(np.delete(mat, 0, axis=0), c, axis=1)
        determinant += ((-1) ** c) * mat[0][c] * det(sub_mat)

    return determinant

def back_solution( mat, b ):
    if len(mat) != len(b): raise TypeError("the 2 argument has to be the same lenght")
    if len(mat) != len(mat[0]): raise ValueError("the matrix has to be square")

    x = []
    for temp in range(len(mat)):
        i = len(mat)-1 - temp
        sum = 0
        for j in range( i , len(mat)-1):
            sum += mat[i,j]*x[j-1]
        x.append( (b[i] - sum)/mat[i,i])

    return x

def gauss( mat , b):
    if len(mat) != len(b): raise TypeError("the 2 argument has to be the same lenght")
    if len(mat) != len(mat[0]): raise ValueError("the matrix has to be square")

    part_pivot( mat , b )

    for i in range( len(mat)-1):
        for j in range( i+1 , len(mat)):
            k = mat[i,i] / mat[j,i]
            mat[j] = k * mat[j] - mat[i]
            b[j] = k * b[j] - b[i]


def part_pivot(mat , b):
    if len(mat) != len(b): raise TypeError("the 2 argument has to be the same lenght")
    if len(mat) != len(mat[0]): raise ValueError("the matrix has to be square")

    tmp_vec = 0
    tmp = 0
    for i in range( len(mat)):
        M = mat[i,i]
        for j in range( i+1 , len(mat)):
            if mat[j,i] > M:
                tmp_vec = mat[i]
                mat[i] = mat[j]
                mat[j] = tmp_vec

                tmp = b[i]
                b[i] = b[j]
                b[j] = tmp

                M = mat[j,i]

if __name__ == '__main__':
    mat = np.array([[2,1,1],[0,1,-2],[0,0,1]] , dtype=np.float32)
    b = np.array( [1,-1,4] )

    rnd_mat = 5*np.random.rand( 3,3)-2.5
    rnd_b = 10*np.random.rand(3) - 5

    rnd_mat = np.array( [[ 1.29829867 , 2.02913483 ,-1.06268165],[-2.35275806 ,-2.24769644 ,-0.02587188],[-1.965434  ,  1.75358977 ,-2.07541666]])
    # rnd_b = np.array( [ 1.54484173 ,-3.42745579 ,-2.6530721 ])
    # print( back_solution(mat , b ))
    # print( rnd_mat , rnd_b)
    # gauss( rnd_mat , rnd_b)
    # print(rnd_mat , rnd_b)
    # part_pivot( rnd_mat , rnd_b)

    # Example usage
    matrix = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
    print(det(matrix))  # Output should be 1

    