import mat
import numpy as np
import matplotlib.pyplot as plt

def runge_func(x:float) -> float:
    return 1/(1+25*np.power(x,2))

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
    return np.array([np.array([x[i]**j for j in range(len(y))], np.float32) for i in range(len(x))] , np.float32)

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
    generate a functio that lets you evaluate the polynomial in a set point using newton's basis

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
    return np.dot( y , mat.inverse_mat(base(x,y)).T )


if __name__ == '__main__':
    x = np.linspace(-1,1,25)
    y = runge_func(x)
    

    new = newton( x, y )
    a = interpolation(x,y)
    plt.plot(x,y , 'o')
    plt.plot(np.linspace(min(x),max(x),100) , new(np.linspace(min(x),max(x),100)))
    plt.plot(np.linspace(min(x),max(x),100) , poly(np.linspace(min(x),max(x),100) , a))
    plt.show()
