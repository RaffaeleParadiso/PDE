import numpy as np
from numba import njit
import module.func as func
import module.graph as graph
import module.constants as const
import module.initial_state as state_i

time_steps = const.TIME_STEPS
L = const.LEN
N = 10#const.N_
D = const.D_
INTV = const.INTV_
dt = const.DT
x = const.X
y = const.Y
dx = const.DX
dy = const.DY
betax = const.BETAX
betay = const.BETAY
alphax = const.ALPHAX
alphay = const.ALPHAY

@njit()
def alternddate_direction_implicit():
    cc = 0
    p = state_i.initial_state_(N, x, y, L)
    p_total = np.zeros((time_steps, N, N))
    for t in range(time_steps):
        step = np.zeros(N)
        temp = np.zeros(N)
        p_new = np.zeros(np.shape(p))
        tt = t*dt
        main = np.ones(N)
        up_diag = np.ones(N)
        low_diag = np.ones(N)
        # implicito su x ed esplicito su y
        for j in range(1, N-1):
            for i in range(N):
                up_diag[i] = 0.5*betax*state_i.g(x[i+1], y[j])
                low_diag[i] = -0.5*betax*state_i.g(x[i-1], y[j])
            for i in range(N):
                step[i] = p[i, j]+0.5*betay*(state_i.f(x[i], y[j+1], tt)*p[i, j+1]-state_i.f(x[i], y[j-1], tt)*p[i, j-1]) +\
                    alphay*0.5*(p[i, j+1]-2*p[i, j]+p[i, j-1])
            temp = func.solve_matrix(N, low_diag[1:], main, up_diag[:N-1], step)
            for i in range(N):
                p_new[i, j] = temp[i]
        # implicito su y esplicito su x
        main = np.ones(N)*(1+alphay)
        for i in range(1, N-1):
            for j in range(N-1):
                up_diag[j] = -0.5*betay*state_i.f(x[i], y[j+1], tt+dt/2)-0.5*alphay
                low_diag[j] = 0.5*betay*state_i.f(x[i], y[j-1], tt+dt/2)-0.5*alphay
            for j in range(N-1):
                step[j] = p_new[i, j]-0.5*betax * \
                    (state_i.g(x[i+1], y[j])*p[i+1, j]-state_i.g(x[i-1], y[j])*p[i-1, j])
            temp = func.solve_matrix(N, low_diag[1:], main, up_diag[:N-1], step)
            for j in range(N-1):
                p[i, j] = temp[j]
        p = state_i.boundary_conditions(N,'absorbing', p)
        p_total[t] = p
    return p_total


# @njit(fastmath=True, cache=True)
def alternate_direction_implicit(N, dt):
    '''Execute the integration via Alternate Direction Implicit for a NxN lattice extension'''
    x, y, dx, p, betax, betay, alphax, alphay = func.parameters_var_dt(N, dt)
    time_steps=round(120/dt)
    p_new = np.zeros(np.shape(p))
    p_total = np.zeros((time_steps, N, N))
    print('Time steps', time_steps)

    for t in range(time_steps):
        if t%(1000)==0: print(f'Exe for Nt: {time_steps}, step: {t}' )
        p = func.boundary_conditions('absorbing', N, p)
        tt = t*dt
        # implicito su x ed esplicito su y
        p_new = func.implicit_x_explicit_y(
            N, tt, dt, p, p_new, x, y, betax, betay, alphax, alphay)
        # implicito su y esplicito su x
        p = func.implicit_y_explicit_x(
            N, tt, dt, p, p_new, x, y, betax, betay, alphax, alphay)
        p_total[t]=p
    return(p_total)

p_total = alternate_direction_implicit(N, dt)
print(p_total[0,:,:])
for i in range(len(p_total)):
    np.savetxt(f"dt005/timesteps_{i}.txt", p_total[i,:,:])
# graph.animate_matplotlib(x, y, p_total, "ADI")
# c = np.loadtxt("dat/c_.txt")
# a = np.loadtxt("dat2/c_{i}.txt")