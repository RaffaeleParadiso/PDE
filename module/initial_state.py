import numpy as np
from numba import njit

@njit()
def initial_state_(N, x, y, L):
    '''Construct the initial state'''
    p_i = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            p_i[i, j] = np.exp(-((x[i])**2+(y[j])**2)/(2))
    return p_i

@njit()
def f(x, y, t):
    w = 2
    a0 = 1
    a = a0*np.cos(w)
    return (x-a*y*(1-x**2))

@njit()
def g(x, y):
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

@njit()
def boundary_conditions(dimension, kind, matrix):
    N=dimension
    if kind == 'reflecting':  #funzionano se non c'è un drift che spinge continuamente la funzione sui bordi
        matrix[0, :] = matrix[1, :]
        matrix[:, 0] = matrix[:, 1]
        matrix[N-1, :] = matrix[N-2, :]
        matrix[:, N-1] = matrix[:, N-2]

    if kind == 'periodic':  # ok
        matrix[0, :] = matrix[N-2, :]
        matrix[:, 0] = matrix[:, N-2]
        matrix[N-1, :] = matrix[1, :]
        matrix[:, N-1] = matrix[:, 1]

    if kind == 'absorbing':  # più o meno
        matrix[0, :] = 0
        matrix[:, 0] = 0
        matrix[N-1, :] = 0
        matrix[:, N-1] = 0
    return matrix