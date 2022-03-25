import numpy as np
from numba import njit

@njit()
def initial_state_(N, x, y, L):
    '''Construct the initial state'''
    p_i = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            p_i[i, j] = np.exp(-((x[i]-0.3*L)**2+(y[j]-0.5*L)**2)/(2))
    return p_i

@njit()
def f(x, y, t):
    w = 0.1
    a0 = 0.05
    a = a0*np.cos(w*t)
    return (x-a*y*(1-x**2))

@njit()
def g(x, y):
    return y

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