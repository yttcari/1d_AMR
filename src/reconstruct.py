import numpy as np

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

def WENO(U, solver, dt, dx, N, X, **kwargs):
    num_vars = U.shape[1]
    NG = 2  # number of ghost cells

    def weno4_reconstruct(U, dx):

        N = len(U)
        UL = np.zeros(N + 1)
        UR = np.zeros(N + 1)
        
        # Add ghost cells
        NG = 2
        U_pad = np.pad(U, NG, mode='edge')
        dx_pad = np.pad(dx, NG, mode='edge')
        
        eps = 1e-6
        
        for i in range(N + 1):
            ip = i + NG
            
            # Get stencil values
            f_m2, f_m1, f_0, f_p1 = U_pad[ip-2:ip+2]
            dx_m2, dx_m1, dx_0 = dx_pad[ip-2:ip+1]
            
            # Right-biased reconstruction (UR[i])
            p0 = (1/3) * f_m2 - (7/6) * f_m1 + (11/6) * f_0
            p1 = -(1/6) * f_m1 + (5/6) * f_0 + (1/3) * f_p1
            
            # Smoothness indicators
            beta0 = (13/12) * (f_m2 - 2*f_m1 + f_0)**2 + (1/4) * (f_m2 - 4*f_m1 + 3*f_0)**2
            beta1 = (13/12) * (f_m1 - 2*f_0 + f_p1)**2 + (1/4) * (f_m1 - f_p1)**2
            
            # Weights
            gamma0, gamma1 = 1/3, 2/3
            alpha0 = gamma0 / (eps + beta0)**2
            alpha1 = gamma1 / (eps + beta1)**2
            
            omega0 = alpha0 / (alpha0 + alpha1)
            omega1 = alpha1 / (alpha0 + alpha1)
            
            UR[i] = omega0 * p0 + omega1 * p1
            
            # Left-biased reconstruction (UL[i])
            # Flip the stencil
            f_p2, f_p1, f_0, f_m1 = U_pad[ip+1:ip-3:-1]
            
            p0 = (1/3) * f_p2 - (7/6) * f_p1 + (11/6) * f_0
            p1 = -(1/6) * f_p1 + (5/6) * f_0 + (1/3) * f_m1
            
            beta0 = (13/12) * (f_p2 - 2*f_p1 + f_0)**2 + (1/4) * (f_p2 - 4*f_p1 + 3*f_0)**2
            beta1 = (13/12) * (f_p1 - 2*f_0 + f_m1)**2 + (1/4) * (f_p1 - f_m1)**2
            
            alpha0 = gamma0 / (eps + beta0)**2
            alpha1 = gamma1 / (eps + beta1)**2
            
            omega0 = alpha0 / (alpha0 + alpha1)
            omega1 = alpha1 / (alpha0 + alpha1)
            
            UL[i] = omega0 * p0 + omega1 * p1
        
        return UL, UR

    UL = np.zeros((N + 1, num_vars))
    UR = np.zeros((N + 1, num_vars))

    for var in range(num_vars):
        UL_recon, UR_recon = weno4_reconstruct(U[:, var], dx)
        UL[:, var] = UL_recon
        UR[:, var] = UR_recon

    flux = np.zeros((N + 1, num_vars))
    for i in range(N + 1):
        flux[i] = solver(UL[i], UR[i]) 

    # Conservative update
    dU = (dt / dx[:, np.newaxis]) * (flux[1:] - flux[:-1])

    return -dU