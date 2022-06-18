import logging
import numpy as np
from numba import njit
import module.func as func
import module.graph as graph
import module.constants as const
import shutil
import os

logging.basicConfig(level=logging.INFO)

cammino = 'vtk_duffing'
if os.path.exists(cammino) == True:
    shutil.rmtree(cammino)

os.makedirs(cammino)
print(f'directory {cammino} created')

time_steps = 40000
L = const.L
N = 100
INTV = 150
cc = 0
dt = 2e-3
D = 0.8

x = np.linspace(-L, L, N)
y = x
p_new = np.zeros((N, N))
p = p_new

dx = x[1]-x[0]
dy = dx
betax = 0.5*dt/dx
betay = 0.5*dt/dy
alphax = dt*D/dx**2
alphay = dt*D/dy**2
logging.info(f'alpha = {alphay}')


@njit()
def initial_state_(N, p, dx, dy):
    '''Construct the initial state'''
    a = 0.5
    for i in range(N):
        for j in range(N):
            p[i, j] = np.exp(-((x[i]-0.2*L)**2+(y[j]-0.2*L)**2)/2*a)
    return p


@njit()
def g(x, y):
    return(y)


@njit()
def f(x, y,t):
    a = 0.1#*np.sin(0.3*t)**2
    w = 0.4*np.cos(0.8*t)+np.sin(0.4*t)
    return(x*(1-a*x**2)-2*w*y)


@njit()
def h(x, y):
    # (x**2*0.2+2*x*0.2+0.2) #ritorna 1 perché gli altri so' parametri senza senso
    return 1

@njit()
def alternate_direction_implicit(p):
    # updating intermediate and final vectors
    p_total = np.zeros((round(time_steps/INTV)+1, N, N))
    p_new = np.zeros(np.shape(p))
    step = np.zeros(N)
    temp = np.zeros(N)

    # tridiagonal construct
    cc=0
    main = np.ones(N)
    up_diag = np.ones(N)
    low_diag = np.ones(N)
    for t in range(time_steps):
        tt=dt*t+dt/2
        for j in range(N-1):
            for i in range(N):
                up_diag[i] = g(x[i+1], y[j])*0.5*betax
                low_diag[i] = -g(x[i-1], y[j])*0.5*betax
            up_diag[N-1]=0
            low_diag[0]=0
            for i in range(N):
                step[i] = p[i, j]-0.5*betay*(f(x[i], y[j+1], tt)*p[i, j+1]-f(x[i], y[j-1], tt)*p[i, j-1]) +\
                    0.25*alphay*(h(x[i], y[j+1])*p[i, j+1]-2 *
                                 h(x[i], y[j])*p[i, j]+h(x[i], y[j-1])*p[i, j-1])

            temp = func.solve_matrix(
                N, low_diag[1:], main, up_diag[:N-1], step)
            for i in range(N):
                p_new[i, j] = temp[i]

        main = np.ones(N)*(1+alphay*0.5)
        for i in range(N-1):
            for j in range(N):
                up_diag[j] = f(x[i], y[j+1], tt)*0.5*betay - \
                    alphay*0.25*h(x[i], y[j+1])
                low_diag[j] = -0.5*f(x[i], y[j-1], tt)*betay - \
                    alphay*0.25*h(x[i], y[j-1])
            up_diag[N-1]=0
            low_diag[0]=0
            for j in range(N):
                step[j] = p[i, j]-(g(x[i+1], y[j])*p[i+1, j] -
                                   g(x[i-1], y[j])*p[i-1, j])*0.5*betax

            temp = func.solve_matrix(
                N, low_diag[1:], main, up_diag[:N-1], step)
            for j in range(N):
                p[i, j] = temp[j]

        # BC periodic
        # p[0, :] = p[N-2]
        # p[:, 0] = p[:, N-2]
        # p[N-1, :] = p[1, :]
        # p[:, N-1] = p[:, 1]

         # BC absorbing
        p[0, :] = 0
        p[:, 0] = 0
        p[N-1, :] = 0
        p[:, N-1] = 0

        if t % INTV == 0:
            p_total[cc] = p
            print(f'time step {t}')
            cc += 1

    return p_total


p = initial_state_(N, p, dx, dy)
p_total = alternate_direction_implicit(p)
graph.animate_matplotlib(x, y, p_total)

for i in range(len(p_total)):
    graph.writeVtk(i, p_total[i], N, dx, cammino)
