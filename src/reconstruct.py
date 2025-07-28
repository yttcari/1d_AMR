import numpy as np
from misc import *

def dx_method(dx_type):
    if dx_type == 'godunov':
        return godunov
    if dx_type == 'MUSCL':
        return MUSCL
    if dx_type == 'PPM':
        return PPM
    else:
        raise ValueError("The reconstruction method is not implemented.")

# ====== Godunov ======
def godunov(U, solver, dt, dx, N, X, **kwargs):
    U, X = generate_gc(U, X, 1) # add ghost cell
    num_vars = U.shape[1]
    flux = np.zeros((N + 1, num_vars))
    for i in range(N + 1):
        flux[i] = solver(U[i], U[i+1])
    dU = (dt / dx[:, np.newaxis]) * (flux[1:] - flux[:-1])

    return -dU

# ===== MUSCL =====

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

def minmod(a, b):
    sigma = np.where(a * b < 0.0, 0.0, np.where(np.abs(a) < np.abs(b), a, b))
    return sigma

def get_slope(U, X):
    slope_p = get_mp(U, X)
    slope_m = get_mm(U, X)

    return minmod(slope_p, slope_m)

def MUSCL(U, solver, dt, dx, N, X, **kwargs):
    num_vars = U.shape[1]
    U, X = generate_gc(U, X, 1) # add ghost cell
    sigma_gc = get_slope(U, X)

    dx_gc = np.pad(dx, (1, 1), 'edge')

    UL_gc = U + 0.5 * dx_gc[:, np.newaxis] * sigma_gc
    UR_gc = U - 0.5 * dx_gc[:, np.newaxis] * sigma_gc

    flux = np.zeros((N + 1, num_vars))

    for i in range(N + 1):
        flux[i] = solver(UL_gc[i], UR_gc[i+1])
    
    dU = (dt / dx[:, np.newaxis]) * (flux[1:] - flux[:-1])

    return -dU

#========= PPM =========

def PPM(U, solver, dt, dx, N, X, **kwargs):
    
    dU = np.zeros_like(U)
    NG = 3
    
    U_extended = np.pad(U, ((NG, NG), (0, 0)), 'edge')
    dx_extended = np.pad(dx, (NG, NG), 'edge')
 
    xi_j = dx_extended[NG-1:N+NG+1][:, np.newaxis]
    xi_jm1 = dx_extended[NG-2:N+NG][:, np.newaxis]
    xi_jp1 = dx_extended[NG:N+NG+2][:, np.newaxis]
    xi_jp2 = dx_extended[NG+1:N+NG+3][:, np.newaxis]

    a_j = U_extended[NG-1:N+NG+1]
    a_jm1 = U_extended[NG-2:N+NG]
    a_jp1 = U_extended[NG:N+NG+2]

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

    U_L = np.zeros_like(U_extended[2:-2])
    U_R = np.zeros_like(U_extended[2:-2])

    U_L[:-1] = a_boundary[:-1]
    U_R[1:] = a_boundary[1:]

    a_j = U_extended[2:-2]
    # DEBUG
    # print(N, U_L.shape, a_boundary.shape, a_j.shape, U_extended.shape)

    # Limiter
    mask1 = (U_R - a_j) * (a_j - U_L) <= 0
    mask2 = (U_R - U_L) * (a_j - 0.5 * (U_L + U_R)) > (1/6) * (U_L - U_R) ** 2
    mask3 = (U_R - U_L) * (a_j - 0.5 * (U_L + U_R)) < -(1/6) * (U_L - U_R) ** 2

    U_L = np.where(mask2, 3 * a_j - 2 * U_R, np.where(mask1, a_j, U_L))
    U_R = np.where(mask3, 3 * a_j - 2 * U_L, np.where(mask1, a_j, U_R))


    a6 = 6.0 * a_j - 3.0 * (U_R + U_L)
    flux = np.zeros((N + 1, U.shape[1]))

    for i in range(N + 1):
        flux[i] = solver(U_L[i], U_R[i+1])
    
    dU = (dt / dx[:, np.newaxis]) * (flux[1:] - flux[:-1])

    return -dU

"""
def weno5_reconstruct_left(u, i, epsilon):

    #5th-order WENO reconstruction for left state at interface i+1/2
    # Three candidate stencils for left reconstruction
    # Stencil 1: u[i-2], u[i-1], u[i]
    u1 = (2*u[i-2] - 7*u[i-1] + 11*u[i]) / 6
    
    # Stencil 2: u[i-1], u[i], u[i+1]
    u2 = (-u[i-1] + 5*u[i] + 2*u[i+1]) / 6
    
    # Stencil 3: u[i], u[i+1], u[i+2]
    u3 = (2*u[i] + 5*u[i+1] - u[i+2]) / 6
    
    # Smoothness indicators
    beta1 = (13/12) * (u[i-2] - 2*u[i-1] + u[i])**2 + (1/4) * (u[i-2] - 4*u[i-1] + 3*u[i])**2
    beta2 = (13/12) * (u[i-1] - 2*u[i] + u[i+1])**2 + (1/4) * (u[i-1] - u[i+1])**2
    beta3 = (13/12) * (u[i] - 2*u[i+1] + u[i+2])**2 + (1/4) * (3*u[i] - 4*u[i+1] + u[i+2])**2
    
    # Linear weights (for 5th order)
    d1, d2, d3 = 1/10, 6/10, 3/10
    
    # Nonlinear weights
    alpha1 = d1 / (epsilon + beta1)**2
    alpha2 = d2 / (epsilon + beta2)**2
    alpha3 = d3 / (epsilon + beta3)**2
    
    alpha_sum = alpha1 + alpha2 + alpha3
    
    w1 = alpha1 / alpha_sum
    w2 = alpha2 / alpha_sum
    w3 = alpha3 / alpha_sum
    
    # WENO reconstruction
    return w1 * u1 + w2 * u2 + w3 * u3


def weno5_reconstruct_right(u, i, epsilon):

    #5th-order WENO reconstruction for right state at interface i+1/2
    # Check bounds to prevent index errors
    if i+3 >= len(u):
        # Fall back to lower order reconstruction at boundaries
        if i+1 < len(u):
            return u[i+1]
        else:
            return u[i]
    
    # Three candidate stencils for right reconstruction
    # Stencil 1: u[i-1], u[i], u[i+1]
    u1 = (2*u[i-1] + 5*u[i] - u[i+1]) / 6
    
    # Stencil 2: u[i], u[i+1], u[i+2]
    u2 = (-u[i] + 5*u[i+1] + 2*u[i+2]) / 6
    
    # Stencil 3: u[i+1], u[i+2], u[i+3]
    u3 = (2*u[i+1] - 7*u[i+2] + 11*u[i+3]) / 6
    
    # Smoothness indicators
    beta1 = (13/12) * (u[i-1] - 2*u[i] + u[i+1])**2 + (1/4) * (u[i-1] - u[i+1])**2
    beta2 = (13/12) * (u[i] - 2*u[i+1] + u[i+2])**2 + (1/4) * (u[i] - 4*u[i+1] + 3*u[i+2])**2
    beta3 = (13/12) * (u[i+1] - 2*u[i+2] + u[i+3])**2 + (1/4) * (3*u[i+1] - 4*u[i+2] + u[i+3])**2
    
    # Linear weights (for 5th order)
    d1, d2, d3 = 3/10, 6/10, 1/10
    
    # Nonlinear weights
    alpha1 = d1 / (epsilon + beta1)**2
    alpha2 = d2 / (epsilon + beta2)**2
    alpha3 = d3 / (epsilon + beta3)**2
    
    alpha_sum = alpha1 + alpha2 + alpha3
    
    w1 = alpha1 / alpha_sum
    w2 = alpha2 / alpha_sum
    w3 = alpha3 / alpha_sum
    
    # WENO reconstruction
    return w1 * u1 + w2 * u2 + w3 * u3

# ============ END OF WENO =============
"""