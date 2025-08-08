import numpy as np
from misc import generate_gc
NG = 1

def godunov(U, solver, dt, dx, N, X, bc_type, **kwargs):
    #U = np.pad(U, ((NG, NG), (0, 0)), 'edge')
    U = generate_gc(U, NG, bc_type)
    num_vars = U.shape[1]
    flux = np.zeros((N + 1, num_vars))
    for i in range(N + 1):
        flux[i] = solver(U[i], U[i+1])
    dU = (1 / dx[:, np.newaxis]) * (flux[1:] - flux[:-1])

    return -dU