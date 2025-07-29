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
    # Credit from pyppm (https://github.com/python-hydro/ppmpy.git) 
    # Modified with vectorised operation

    dU = np.zeros_like(U)
    NG = 3
    N = U.shape[0]
    q = np.pad(U, ((NG, NG), (0, 0)), 'edge')
    dx_extended = np.pad(dx, (NG, NG), 'edge')

    cs = get_wavespeed(q)

    def construct_parabola(a_data):
        am = np.zeros_like(a_data)
        ap = np.zeros_like(a_data)
        a6 = np.zeros_like(a_data)

        ib = NG - 2
        ie = N + NG

        da0 = np.zeros_like(a_data)
        dap = np.zeros_like(a_data)

        #print(a_data[ib+1:ie+2].shape, a_data[ib-1:ie].shape, a_data[ib+2:ie+3].shape, a_data[ib:ie+1].shape, a_data.shape[0])

        da0[ib:ie+1, :] = 0.5 * (a_data[ib+1:ie+2, :] - a_data[ib-1:ie, :])
        dap[ib:ie+1, :] = 0.5 * (a_data[ib+2:ie+3, :] - a_data[ib:ie+1, :])

        dl = np.zeros_like(a_data)
        dr = np.zeros_like(a_data)
        dr[ib:ie+1] = a_data[ib+1:ie+2] - a_data[ib:ie+1]
        dl[ib:ie+1] = a_data[ib:ie+1] - a_data[ib-1:ie]

        da0 = np.where(dl * dr < 0, 0.0,
                           np.sign(da0) * np.minimum(np.abs(da0),
                                                     2.0 * np.minimum(np.abs(dl),
                                                                      np.abs(dr))))

        dl[:, :] = dr[:, :]
        dr[ib:ie+1, :] = a_data[ib+2:ie+3, :] - a_data[ib+1:ie+2, :]

        dap = np.where(dl * dr < 0, 0.0,
                           np.sign(dap) * np.minimum(np.abs(dap),
                                                     2.0 * np.minimum(np.abs(dl),
                                                                      np.abs(dr))))

        aint = np.zeros_like(a_data)
        aint[ib:ie+1, :] = 0.5 * (a_data[ib:ie+1, :] + a_data[ib+1:ie+2, :]) - \
                             (1.0 / 6.0) * (dap[ib:ie+1, :] - da0[ib:ie+1, :])

        ap[:, :] = aint[:, :]
        am[1:, :] = ap[:-1, :]

        test = (ap - a_data) * (a_data - am) < 0
        da = ap - am
        testm = da * (a_data - 0.5 * (am + ap)) > da**2 / 6
        am[:, :] = np.where(test, a_data, np.where(testm, 3.0*a_data - 2.0*ap, am))

        testp = -da**2 / 6 > da * (a_data - 0.5 * (am + ap))
        ap[:, :] = np.where(test, a_data, np.where(testp, 3.0*a_data - 2.0*am, ap))
        
        a6[:, :] = 6.0 * a_data - 3.0 * (am + ap)

        return am, ap, a6

    def integrate_parabola(sigma, am, ap, a6):
        Ip = np.where(sigma <= 0.0, ap,
                         ap - 0.5 * np.abs(sigma) *
                           (ap - am - (1.0 - (2.0/3.0) * np.abs(sigma)) * a6))

        Im = np.where(sigma >= 0.0, am,
                         am + 0.5 * np.abs(sigma) *
                           (ap - am + (1.0 - (2.0/3.0) * np.abs(sigma)) * a6))
        return Im, Ip
    
    q_am, q_ap, q_a6 = construct_parabola(q)

    Im = np.zeros((q.shape[0], 3, q.shape[1]))
    Ip = np.zeros((q.shape[0], 3, q.shape[1]))

    for iwave, sgn in enumerate([-1, 0, 1]):
        sigma = (q[:, 1] + sgn * cs) * dt / dx_extended
        Im[:, iwave, :], Ip[:, iwave, :] = integrate_parabola(sigma[:, np.newaxis], q_am, q_ap, q_a6)


    q_left_full = np.zeros((N + 2 * NG + 1, U.shape[1]))
    q_right_full = np.zeros((N + 2 * NG + 1, U.shape[1]))
    
    q_ref_m = Im[NG-1:N+NG+1, 0, :] 
    q_ref_p = Ip[NG-1:N+NG+1, 2, :] 

    ev_m, lvec_m, rvec_m = eigen(q_ref_m) 
    ev_p, lvec_p, rvec_p = eigen(q_ref_p)

    dq_m = q_ref_m[:, np.newaxis, :] - Im[NG-1:N+NG+1, :, :]  
    dq_p = q_ref_p[:, np.newaxis, :] - Ip[NG-1:N+NG+1, :, :]  

    beta_xm = np.einsum('mij,mjk->mik', lvec_m, dq_m.transpose(0, 2, 1)).diagonal(axis1=1, axis2=2)
    beta_xp = np.einsum('mij,mjk->mik', lvec_p, dq_p.transpose(0, 2, 1)).diagonal(axis1=1, axis2=2)

    q_right = q_ref_m.copy()
    q_left  = q_ref_p.copy()

    mask_m = ev_m <= 0  
    mask_p = ev_p >= 0 

    for iwave in range(3):
        q_right -= (mask_m[:, iwave][:, np.newaxis] * beta_xm[:, iwave][:, np.newaxis] * rvec_m[:, iwave, :])
        q_left  -= (mask_p[:, iwave][:, np.newaxis] * beta_xp[:, iwave][:, np.newaxis] * rvec_p[:, iwave, :])

    q_right_full[NG-1:N+NG+1, :] = q_right
    q_left_full[NG:N+NG+2, :] = q_left

    U_L = q_left_full[NG:N+NG+1, :]
    U_R = q_right_full[NG:N+NG+1, :]

    flux = np.zeros((N + 1, U.shape[1]))

    for i in range(N + 1):
        flux[i] = solver(U_L[i], U_R[i])
    
    dU = (dt / dx[:, np.newaxis]) * (flux[1:] - flux[:-1])

    return -dU