import numpy as np
from numba import njit, float64, int32
import module.graph as graph
import module.constants as const
import module.initial_state as state_i

time_steps = const.TIME_STEPS
L = const.LEN
N = const.N_
D = const.D_
INTV = const.INTV_
cc = const.CC
dt = const.DT
x = const.X
y = const.Y
dx = const.DX
dy = const.DY
betax = const.BETAX
betay = const.BETAY
alphax = const.ALPHAX
alphay = const.ALPHAY

p = np.zeros((N, N))
p_della_vita = np.zeros((round(time_steps/INTV)+1, N, N))
p = state_i.initial_state_(N, p, x, y, L)
p_new = np.copy(p)

@njit(float64[:,:](float64[:,:], float64[:,:], int32), fastmath=True, cache=True)
def calcolo(p, p_new, tt):
    for i in range(1, N-1):
        for j in range(1, N-1):
            p_new[i, j] = p[i, j]-betax*(state_i.g(x[i+1], y[j])*p[i+1, j] +\
                            -state_i.g(x[i-1], y[j])*p[i-1, j]) +\
                            betay*(state_i.f(x[i], y[j+1], tt)*p[i, j+1] +\
                            -state_i.f(x[i], y[j-1], tt)*p[i, j-1]) +\
                            alphay*(p[i, j+1]-2*p[i, j]+p[i, j-1])
    return p_new

cc = 0
for t in range(time_steps):
    tt = dt*t
    p = calcolo(p, p_new, tt)
    p_new = state_i.boundary_conditions(N, 'absorbing', p_new)
    p=state_i.boundary_conditions(N, 'absorbing', p)
    p = p_new
    if t % INTV == 0:
        p_della_vita[cc, :, :] = p
        cc += 1 
graph.animate_matplotlib(x, y, np.array(p_della_vita), "explicit")
