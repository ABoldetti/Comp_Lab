import mat
import numpy as np
import matplotlib.pyplot as plt

def runge_func(x:float) -> float:
    return 1/(1+25*np.power(x,2))

def chebyshev_nodes(n:int) -> np.array:
    '''
    generates n chebyshev nodes in the interval [-1,1]
    
    Parameters
    ----------
    n: int
        number of nodes
    
    Returns
    -------
    np.array
        chebyshev nodes'''
    return np.array([-np.cos(i*np.pi/(n-1)) for i in range(n)])

def vandermonde_matrix(x:np.array , y:np.array) -> np.matrix:
    '''
    generates the vandermonde matrix of the system of equations

    Parameters
    ----------
    x : np.array
        x values
    y : np.array
        y values
    
    Raises
    ------
    1: len(x) != len(y)
        two vectors has to have the same lenght
    
    Returns
    -------
    matrix: np.matrix
        the vandermonde matrix of the system of equations

    
    '''
    if len(x) != len(y): raise ValueError('x and y must have the same length')
    return np.array([np.array([np.power(x[i] , j) for j in range(len(y))], np.float64) for i in range(len(x))] , np.float64)

def poly(x:np.array , a:np.array) -> np.array:
    '''
    evaluate interpolation on a polynomial basis
    
    Parameters
    ----------
    x: np.array
        interpolation points
    a: np.array
        coefficients
    
    Returns
    --------
    var: np.array
        points valued'''
    return np.array([sum([a[j]*x[i]**j for j in range(len(a))]) for i in range(len(x))])

def newton( x:np.array , y:np.array):
    '''
    generate a function that lets you evaluate the polynomial in a set point using newton's basis

    Parameters
    ----------
    x : np.array
        x values
    y : np.array
        y values
    
    Raises
    ------
    1: len(x) != len(y)
        two vectors has to have the same lenght
    
        
    Returns
    --------
    eval_poly: function
        function that given an array of floats or a sigle point, returns the value of the function interpolated in that point
    
    '''

    mat = np.zeros((len(x) , len(x)) , dtype= np.float32)
    for i in range(len(y)):
        mat[i,0] = y[i]
    for i in range( 1, len(x) ):
        for j in range( len(x) - i ):
            mat[j,i] = (mat[j+1 , i-1] - mat[j , i-1])/ (x[j+i] - x[j])
    
    
    a = np.array([ mat[0,i] for i in range(len(x))])

    def eval_poly( x0: float):
        '''
        function for the evaluation of the interpolation in one point
        
        Parameters
        ----------
        x0: float or np.array
            evaluating point
        
        Returns
        -------
        var: float or np.array
            function evaluated
        '''
        if isinstance(x0 , (int,float)):
            return sum([a[i]* np.prod([ x0 - x[j] for j in range(i)]) for i in range(len(x))])
        if isinstance(x0 , (np.ndarray , list)):
            return np.array([sum(np.array([a[i]* np.prod(np.array([ x0[k] - x[j] for j in range(i)]) , axis = 0) for i in range(len(x))])) for k in range(len(x0))])
    return eval_poly
    
def interpolation( x , y , base = vandermonde_matrix):
    '''
    interpolation function

    Parameters
    ----------
    x : np.array
        x values
    y : np.array
        y values
    base : function
        base function that generates the matrix of the system of equations
        default is vandermonde_matrix
    
    Raises
    ------
    1: len(x) != len(y)
        two vectors has to have the same lenght

    Returns
    -------
    np.array
        coefficients of the polynomial written in the same basis
    '''

    if len(x) != len(y): raise ValueError('x and y must have the same length')
    return mat.solve_mat(base(x,y) , y)


def continuity( m , x , deg , line , col , last):
    '''
    function that generates the continuity constraints on the spline matrix
    
    Parameters
    ----------
    m : np.array
        matrix of the system of equations
    x : float
        x 
    deg : int
        degree of the polynomial
    line : int
        line of the matrix
    col : int
        column of the matrix
        
    if a matrix is [[1,2,3],[4,5,6],[7,8,9]], the number 4 is in the second line and first column
    matrix[1,0] = 4
        
        '''
    
    for i in range(deg+1):
        m[line + i , col] = np.power(x , i)
        if not last:
            m[line + i , col+deg] = np.power(x , i)

def derivability( m , x , deg , line ,col , last):
    for i in range(deg+1):
        
    
    pass


def splines( x:np.array , y:np.array , deg:int):
    '''
    spline interpolation function

    Parameters
    ----------
    x : np.array
        x values
    y : np.array
        y values
    deg: int
        degree of interpolation
    Raises
    ------
    1: len(x) != len(y)
        two vectors has to have the same lenght
    2: deg<0
        degree of polynomail can't be negative
    3: type(deg) != int
        degree can't be non int

    Returns
    -------
    np.array
        coefficients of the polynomial written in the same basis
    '''
    if len(x) != len(y): raise ValueError('x and y must have the same length')
    if deg <0: raise ValueError('degree of polynomial cannot be negative')
    if not isinstance(deg, int): raise TypeError('deg has to be int')
    
    len_mat = (deg+1) * (len(x)-1)

    m = np.zeros((len_mat , len_mat) , dtype= np.float32)
    known_vec = np.copy(y)
    for i in range(len(y)-1 , 0 , -1):
        known_vec = np.insert(known_vec , i , y[i])
    known_vec = np.concatenate( known_vec , np.zeros(((deg+1) * (len(x)-1)) - len(known_vec) , dtype= np.float32))
    
    for i in range(0,len(x)-1):
        continuity(m , x[i+1] , deg , 1+(2*i) ,deg*i , not bool(i-len(x-2)))
    
    for j in range( 1 , len(x)-1):
        derivability( m , x[i+1] , deg , 1+(2*i) ,deg*i , not bool(i-len(x-2)))
        

    



    
    
    [1,2,3,4,5]


    '''
    NOTE: Spline
        matrix (n of coefficients x n of intervals) = ((n deg Poly + 1 ) x (n points - 1))
        Una volta che ho la matrice risolvo il sistema M*coeff = y, trovo i coefficienti 
        Avuti i coefficienti li uso per funzioni del tipo Sum( ai*x ) con x da far variare in un intervallo [x[i] , x[i+1]]
    '''
    


if __name__ == '__main__':
    #x = np.linspace(-1,1,25)
    x = chebyshev_nodes(5)
    y = runge_func(x)
    

    new = newton( x, y )
    a = interpolation(x,y)
    plt.plot(x,y , '-')
    #plt.plot(np.linspace(min(x),max(x),100) , new(np.linspace(min(x),max(x),100)))
    # plt.plot(np.linspace(min(x),max(x),100) , poly(np.linspace(min(x),max(x),100) , a))
    plt.show()

    