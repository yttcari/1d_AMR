import numpy as np
from misc import generate_gc, minmod

NG = 1

def van_leer(a, b):
    return np.where((a * b) <= 0.0, 0.0, (2 * a * b) / (a + b + 1e-12))

def mc_limiter(a, b):
    return np.where(a * b <= 0, 0.0, np.sign(a) * np.minimum(np.abs(a + b) / 2, np.minimum(2*np.abs(a), 2*np.abs(b))))


def generate_X_gc(X, NG):
    N = len(X)
    X_with_gc = np.zeros(N + NG * 2)

    X_with_gc[NG:-NG] = X

    dx_left = X_with_gc[NG+1] - X_with_gc[NG]
    dx_right = X_with_gc[-NG-1] - X_with_gc[-NG-2]

    for i in range(NG):
        X_with_gc[NG - 1 - i] = X_with_gc[NG - i] - dx_left # Left ghost cells
        X_with_gc[NG + N + i] = X_with_gc[NG + N + i - 1] + dx_right # Right ghost cells

    return X_with_gc

def get_mp(U_arr, X_arr):
    dX = X_arr[1:] - X_arr[:-1]
    # get slope from forward differencing
    slope_plus = np.zeros_like(U_arr)
    slope_plus[:-1] = (U_arr[1:] - U_arr[:-1]) / dX[:, np.newaxis]

    return slope_plus

def get_mm(U_arr, X_arr):
    dX = X_arr[1:] - X_arr[:-1]
    # get slope from backward differencing
    slope_minus = np.zeros_like(U_arr)
    slope_minus[1:] = (U_arr[1:] - U_arr[:-1]) / dX[:, np.newaxis]

    return slope_minus

def get_slope(U, X):
    s = minmod(get_mp(U, X))
    t = minmod(get_mm(U, X))

    return np.where(s * t < 0.0, 0.0, np.where(np.abs(s) < np.abs(t), s, t))

def MUSCL(U, solver, dt, dx, N, X, bc_type='outflow', **kwargs):
    num_vars = U.shape[1]
    #U = np.pad(U, ((NG, NG), (0, 0)), 'edge')
    U = generate_gc(U, NG, bc_type)
    X = generate_X_gc(X, 1) # add ghost cell

    #print(U_ext.shape, X_ext.shape, U.shape, X.shape)
    sigma_gc = get_slope(U, X)

    #dx_gc = np.pad(dx, (1, 1), 'edge')
    dx_gc = generate_gc(dx, 1, bc_type)

    UL_gc = U + 0.5 * dx_gc[:, np.newaxis] * sigma_gc
    UR_gc = U - 0.5 * dx_gc[:, np.newaxis] * sigma_gc

    flux = np.zeros((N + 1, num_vars))

    for i in range(N + 1):
        flux[i] = solver(UL_gc[i], UR_gc[i+1])
    
    dU = (dt / dx[:, np.newaxis]) * (flux[1:] - flux[:-1])

    return -dU