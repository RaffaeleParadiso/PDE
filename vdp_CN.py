import logging
import os
import shutil
import numpy as np
from numba import njit, float64, int64
import module.func as func
import module.graph as graph
import module.constants as const

logging.basicConfig(level=logging.INFO)

time_steps = const.TIME_STEPS-10000
L = const.L-2
N = const.N
INTV=const.INTV
cc=const.cc
dt = const.dt
D = const.D
execute=const.execute

x = np.linspace(-L, L, N)
y=np.linspace(-L, L, N)
p_new = np.zeros((N, N))
p = np.zeros((N, N))
p_total =np.zeros((round(time_steps/INTV)+10, N, N))

dx = (x[1]-x[0])
dy = y[1]-y[0]
betax = dt/dx
betay = dt/dy
alphax = dt*D/dx**2
alphay = dt*D/dy**2
logging.info(f'alphax = {alphax}')

# initial state
p, p_new=func.initial_state_(N, p, p_new, dx, dy, L)

@njit(float64[:,:](float64[:,:], float64[:,:], int64), fastmath=True, cache=True)
def implicit_calcx(p, p_new, tt):
    N=const.N
    main_diag_x = np.ones(N)
    diag_low_x = np.ones(N-1)
    diag_up_x = np.ones(N-1)
    for j in range(N):
        for i in range(N):
            diag_low_x[i] = (-betax*func.g(x[i-1], y[j]))*0.5
            diag_up_x[i] = (betax*func.g(x[i+1], y[j]))*0.5
        for i in range(N):
            p_new[i, j]=-betax*0.5*(func.g(x[i+1], y[j])*p[i+1, j]-func.g(x[i-1], y[j])*p[i-1, j])+p[i, j]
        p_new[:, j] = func.solve_matrix(N, diag_low_x, main_diag_x, diag_up_x, p[:, j])
    return p_new

@njit(float64[:,:](float64[:,:], float64[:,:], int64), fastmath=True, cache=True)
def implicit_calcy(p, p_new, tt):
    N=const.N
    main_diag_y = np.ones(N)*(1+2*alphay)
    diag_low_y = np.ones(N-1)
    diag_up_y = np.ones(N-1)
    for i in range(N):
        for j in range(N):
            diag_low_y[j] = (betay*func.f(x[i], y[j-1], tt)-alphay)*0.5
            diag_up_y[j] = (-betay*func.f(x[i], y[j+1], tt)-alphay)*0.5
        for j in range(N):
            p[i, j]=(betay*(func.f(x[i], y[j+1], tt)*p_new[i, j+1]-func.f(x[i], y[j-1], tt)*p_new[i, j-1])+0.5*alphay*(p_new[i, j+1]-\
                        2*p_new[i, j]+p_new[i, j-1]))*0.5+p_new[i, j]
        p[i, :] = func.solve_matrix(N, diag_low_y, main_diag_y, diag_up_y, p_new[i, :])
    return p

if execute==True:
    for t in range(time_steps):
        tt = dt*t
        p_new = implicit_calcx(p, p_new, tt)
        p = implicit_calcy(p, p_new, tt)
        p=func.boundary_conditions(N, 'absorbing', p)
        p_new=func.boundary_conditions(N, 'absorbing', p_new)
        if t % INTV == 0:
            logging.info(f'step {cc}')
            p_total[cc, :, :] = p
            # func.writeVtk(cc, p, N, dx, cammino)
            cc += 1
    graph.animate_matplotlib(x, y, p_total)
