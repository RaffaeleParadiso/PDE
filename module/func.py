import numpy as np
from numba import njit, float64, int32

@njit(float64[:](int32, float64[:], float64[:], float64[:], float64[:]), fastmath=True, cache=True) #questa funzione risolve il sistema di equazioni invece di invertire matrici
def solve_matrix(n, lower_diagonal, main_diagonal, upper_diagonal, solution_vector):

    '''Solve systems of equations instead of inverting matrices. It returns
       the same solution of np.linalg.solve'''

    w=np.zeros(n-1)
    g=np.zeros(n)
    result=np.zeros(n)

    w[0]=upper_diagonal[0]/main_diagonal[0]
    g[0]=solution_vector[0]/main_diagonal[0]

    for i in range(1, n-1):
        w[i]=upper_diagonal[i]/(main_diagonal[i]-lower_diagonal[i-1]*w[i-1])
    for i in range(1, n):
        g[i]=(solution_vector[i]-lower_diagonal[i-1]*g[i-1])/(main_diagonal[i]-lower_diagonal[i-1]*w[i-1])
    result[n-1]=g[n-1]
    for i in range(n-1, 0, -1):
        result[i-1]=g[i-1]-w[i-1]*result[i]
    return result #restituisce la stessa soluzione che con linalg.solve

def tridiag_construct(a, b, c, k1=-1, k2=0, k3=1):
    '''Construction of tridiagonal matrices'''
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

def boundary_conditions(dimension, kind, vector):
    '''
    Impose the boundary conditions, kind should be "reflecting", "absorbing" or "periodic" 
    '''
    N=dimension
    if kind == 'reflecting':  #funzionano se non c'è un drift che spinge continuamente la funzione sui bordi
        vector[0, :] = vector[1, :]
        vector[:, 0] = vector[:, 1]
        vector[N-1, :] = vector[N-2, :]
        vector[:, N-1] = vector[:, N-2]

    if kind == 'periodic':  # ok
        vector[0, :] = vector[N-2, :]
        vector[:, 0] = vector[:, N-2]
        vector[N-1, :] = vector[1, :]
        vector[:, N-1] = vector[:, 1]

    if kind == 'absorbing':  # più o meno
        vector[0, :] = 0
        vector[:, 0] = 0
        vector[N-1, :] = 0
        vector[:, N-1] = 0
    return vector

def initial_state_(N, p, p_new, dx, dy, L):
    '''Construct the initial state'''
    for i in range(N):
        xi = dx*i
        for j in range(N):
            yj = dy*j
            p[i, j] = np.exp(-((xi-L)**2+(yj-L)**2)/0.3)
            p_new[i, j] = p[i, j]
    return p, p_new


@njit()
def f(x, y, t):
    '''drift function f for Van Der Pol'''
    a = 0.1#0.2+0.2*np.cos(0.2*t)
    b = 0.1
    return(x-a*y*(1-x**2)) #(x-a*y*(1-b*x**2+0.1*0.2)-0.1*0.2)

@njit()
def g(x, y):
    '''drift function g for Van der Pol'''
    return y

@njit()
def f_duff(x, y, t):
    '''drift function f for Duffing oscillator'''
    return (x*(1-0.1*x**2)-2*0.05*y)

@njit()
def g_duff(x, y):
    '''drift function g for Duffing oscillator'''
    return y

@njit()
def f_parisi(x, y, t):
    '''Drift function f for Parisi FP'''
    w=1e-5
    a = 1+0.0005*np.cos(w*t)
    return -x+x**3+a*x+y

@njit()
def g_parisi(x, y):
    '''Drift function g for Parisi FP'''
    tau=0.05
    return y/tau