import numpy as np
from FVM import con2prim, prim2con, GAMMA

def generate_gc(U, X, NG, order=0):
    N = U.shape[0]
    num_vars = U.shape[1] if U.ndim > 1 else 1

    data_with_gc = np.zeros((N + 2 * NG, num_vars)) if num_vars > 1 else np.zeros(N + 2 * NG)
    X_with_gc = np.zeros(N + 2 * NG)

    data_with_gc[NG:-NG] = U
    X_with_gc[NG:-NG] = X

    if order == 0:
        for i in range(NG):
            if num_vars > 1:
                data_with_gc[i, :] = data_with_gc[NG, :]
                data_with_gc[-(i + 1), :] = data_with_gc[-NG - 1, :]
            else:
                data_with_gc[i] = data_with_gc[NG]
                data_with_gc[-(i + 1)] = data_with_gc[-NG - 1]

        dx_left = X_with_gc[NG+1] - X_with_gc[NG]
        dx_right = X_with_gc[-NG-1] - X_with_gc[-NG-2]

        for i in range(NG):
            X_with_gc[NG - 1 - i] = X_with_gc[NG - i] - dx_left # Left ghost cells
            X_with_gc[NG + N + i] = X_with_gc[NG + N + i - 1] + dx_right # Right ghost cells

    return data_with_gc, X_with_gc

def godunov(U, solver, dt, dx, N, X, **kwargs):
    U, X = generate_gc(U, X, 1) # add ghost cell
    num_vars = U.shape[1]
    flux = np.zeros((N + 1, num_vars))
    for i in range(N + 1):
        flux[i] = solver(U[i], U[i+1])
    dU = (dt / dx[:, np.newaxis]) * (flux[1:] - flux[:-1])

    return -dU

def MUSCL(U, solver, dt, dx, N, X, **kwargs):
    num_vars = U.shape[1]
    U, X = generate_gc(U, X, 1) # add ghost cell

    def minmod(U_arr, X_arr):
        dX = X_arr[1:] - X_arr[:-1]

        slope_minus = np.zeros_like(U_arr)
        slope_plus = np.zeros_like(U_arr)

        slope_minus[1:] = (U_arr[1:] - U_arr[:-1]) / dX[:, np.newaxis]
        
        slope_plus[:-1] = (U_arr[1:] - U_arr[:-1]) / dX[:, np.newaxis]

        sigma = np.where(slope_minus * slope_plus < 0.0, 0.0,
                         np.where(np.abs(slope_minus) < np.abs(slope_plus), slope_minus, slope_plus))
        return sigma
    
    sigma_gc = minmod(U, X)

    dx_gc = np.zeros(N + 2)
    dx_gc[0] = dx[0]         # dx for the left ghost cell
    dx_gc[1:N+1] = dx        # dx for the N computational cells
    dx_gc[N+1] = dx[-1]      # dx for the right ghost cell

    UL_gc = U + 0.5 * dx_gc[:, np.newaxis] * sigma_gc
    UR_gc = U - 0.5 * dx_gc[:, np.newaxis] * sigma_gc

    flux = np.zeros((N + 1, num_vars))

    for i in range(N + 1):
        flux[i] = solver(UL_gc[i], UR_gc[i+1])
    
    dU = (dt / dx[:, np.newaxis]) * (flux[1:] - flux[:-1])

    return -dU

#========= PPM =========

def PPM(U, solver, dt, dx, N, X, **kwargs):
    
    def minmod_limiter(a, b):
        if a * b <= 0:
            return 0.0
        elif abs(a) < abs(b):
            return a
        else:
            return b
    
    dU = np.zeros_like(U)
    NG = 2
    
    U_extended = np.pad(U, ((NG, NG), (0, 0)), 'edge')
    dx_extended = np.pad(dx, (NG, NG), 'edge')

    delta = np.zeros((N + 4, 3))
    for i in range(1, N + 3):
        for k in range(3):
            delta[i, k] = minmod_limiter(
                    (U_extended[i, k] - U_extended[i-1, k]) / dx_extended[i-1],
                    (U_extended[i+1, k] - U_extended[i, k]) / dx_extended[i]
                ) * min(dx_extended[i-1], dx_extended[i])
    """        
    xi_j = dx_extended[NG-1:N+NG][:, np.newaxis]
    xi_jm1 = dx_extended[NG-2:N+NG-1][:, np.newaxis]
    xi_jp1 = dx_extended[NG:N+NG+1][:, np.newaxis]
    xi_jp2 = dx_extended[NG+1:N+NG+2][:, np.newaxis]

    a_j = U_extended[NG-1:N+NG]
    a_jm1 = U_extended[NG-2:N+NG-1]
    a_jp1 = U_extended[NG:N+NG+1]

    def get_daj(xi_jm1, xi_j, xi_jp1, a_jm1, a_j):
        daj = (xi_j / (xi_jm1 + xi_j + xi_jp1) * (
            (2 * xi_jm1 + xi_j) * (a_j - a_jm1) / (xi_jp1 + xi_j) +
            (xi_j + 2 * xi_jp1) * (a_j - a_jm1) / (xi_jm1 + xi_j)
        ))

        return daj

    d_aj = get_daj(xi_jm1, xi_j, xi_jp1, a_jm1, a_j)
    d_ajp1 = get_daj(xi_j, xi_jp1, xi_jp2, a_j, a_jp1)

    a_boundary = (a_j + xi_j / (xi_j + xi_jp1) * (a_jp1 - a_j) + 
                  1 / (xi_jm1 + xi_j + xi_jp1 + xi_jp2) * (
                      2 * xi_jp2 * xi_j * (a_jp1 - a_j) / (xi_j + xi_jp1) * 
                      ((xi_jm1 + xi_j) / (2 * xi_j + xi_jp1) - (xi_jp2 + xi_jp1) / (2 * xi_jp1 + xi_j)) -
                      xi_j * d_ajp1 * (xi_jm1 + xi_j) / (2 * xi_j + xi_jp1) + xi_jp1 * (xi_jp1 + xi_jp2) * d_aj / (xi_j + 2 * xi_jp1)
                  ))
    #print(a_boundary.shape)
    """
    U_L = np.zeros((N + 3, 3))  # Left state at interface i+1/2
    U_R = np.zeros((N + 3, 3))  # Right state at interface i+1/2

    U_L[:, :] = U_extended[:-1, :] + 0.5 * delta[:-1, :]
    U_R[:, :] = U_extended[1:, :] - 0.5 * delta[1:, :]

    #U_L[:, :] = a_boundary[:-1]
    #U_R[:, :] = a_boundary[1:]

    mask1 = (U_L - U_extended[:-1, :]) * (U_extended[:-1, :] - U_R) <= 0
    
    U_L[mask1] = U_extended[:-1, :][mask1]
    U_R[mask1] = U_extended[:-1, :][mask1]

    mask2 = (U_L - U_extended[:-1, :]) * (U_extended[1:, :] - U_R) <= 0

    mask_cond1 = mask2 & (np.abs(U_L - U_extended[:-1, :]) >= 2 * np.abs(U_R - U_extended[1:, :]))
    U_L[mask_cond1] = U_extended[:-1, :][mask_cond1] - 2 * (U_R[mask_cond1] - U_extended[1:, :][mask_cond1])

    mask_cond2 = mask2 & (np.abs(U_R - U_extended[1:, :]) >= 2 * np.abs(U_L - U_extended[:-1, :]))
    U_R[mask_cond2] = U_extended[1:, :][mask_cond2] - 2 * (U_L[mask_cond2] - U_extended[:-1, :][mask_cond2])

    F = np.zeros((N + 1, 3))
    
    for i in range(N + 1):
        
        UL = U_L[i+1, :]
        UR = U_R[i+1, :]
        
        # Compute flux using Riemann solver
        F[i, :] = solver(UL, UR)
    dU = -dt/dx[:, np.newaxis] * (F[1:] - F[:-1])
    
    return dU

# ========= END OF PPM =========

def dx_method(type, **kwargs):
    if type == 'godunov':
        return godunov(**kwargs)
    if type == 'MUSCL':
        return MUSCL(**kwargs)
    if type == 'WENO':
        return WENO(**kwargs)
    if type == 'PPM':
        return PPM(**kwargs)
    
def dt_method(dt_type, dx_type, U, **kwargs):
    if dt_type == 'euler':
        dU = dx_method(dx_type, U=U, **kwargs)
        return dU
    elif dt_type == 'rk4':
        k1 = dx_method(dx_type, U=U, **kwargs)
        k2 = dx_method(dx_type, U=U + 0.5 * k1, **kwargs)
        k3 = dx_method(dx_type, U=U + k2 * 0.5, **kwargs)
        k4 = dx_method(dx_type, U=U + k3, **kwargs)
        
        return (k1 + 2 * k2 + 2 * k3 + k4) / 6