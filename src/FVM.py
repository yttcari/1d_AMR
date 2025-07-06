import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
    
CFL = 0.5   
GAMMA = 1.4

############### Global Variable ###############
def get_gamma():
    return GAMMA

def get_CFL():
    return CFL

def update_CFL(new_CFL):
    global CFL

    CFL = new_CFL

def update_GAMMA(new_GAMMA):
    global GAMMA

    GAMMA = new_GAMMA

############### Parameter Update ###############
def prim2con(prim):
    """
    Input: primitive variable [rho, u, p]
    Output: consered variable [rho, rhou, E]
    """

    rho = prim[0]
    u = prim[1]
    p = prim[2]

    E = p / (GAMMA - 1) + 0.5 * rho * (u ** 2)

    return np.array([rho, rho * u, E])

def con2prim(con):
    """
    Input: consered variable [rho, rhou, E]
    Output: primitive variable [rho, u, p]
    """
    rho = con[0]
    # Add checks for rho being close to zero or negative
    if np.any(rho <= 0):
        print(f"Warning: Negative or zero density encountered! rho={rho}")
    
        rho = np.finfo(float).eps # smallest representable positive float

    u = con[1] / rho
    p = (con[2] - 0.5 * rho * (u ** 2)) * (GAMMA - 1)

    # Add checks for p being close to zero or negative
    if np.any(p <= 0):
        print(f"Warning: Negative or zero pressure encountered! p={p} at rho={rho}, u={u}")
        p = np.finfo(float).eps # smallest representable positive float

    return np.array([rho, u, p])

def con2prim_grid(U_values):
    # Ensure U_values is treated as a float NumPy array
    U_values = np.asarray(U_values, dtype=float)

    # Initialize prim with the correct shape and data type
    prim = np.zeros_like(U_values, dtype=float)

    for i, u in enumerate(U_values):
        rho = u[0]
        mom = u[1]
        energy = u[2]
        
        # Robustness for numerical errors
        if rho <= 0: rho = 1e-10
        u_vel = mom / rho
        internal_energy = energy - 0.5 * rho * u_vel**2
        if internal_energy < 0: 
            internal_energy = 1e-10

        pressure = internal_energy * (GAMMA - 1)
        
        # MODIFIED LINE: Assign a NumPy array directly to the row
        prim[i] = np.array([rho, u_vel, pressure], dtype=float) 
        
    return prim

def prim2con_grid(con):
    tmp = []
    for i in con:
        tmp.append(prim2con(i))

    return np.array(tmp)

############### Solver ###############

def HLL_flux(UL, UR):
    # Convert to primitive variables
    rhoL, uL, pL = con2prim(UL)
    rhoR, uR, pR = con2prim(UR)

    # Shorthands for energy
    EL = UL[2]
    ER = UR[2]

    # Compute speed of sound
    cL = np.sqrt(GAMMA * pL / rhoL)
    cR = np.sqrt(GAMMA * pR / rhoR)

    # Estimate wave speeds
    SL = min(uL - cL, uR - cR)
    SR = max(uL + cL, uR + cR)

    # Compute fluxes for left and right states
    FL = np.array([
        rhoL * uL,
        rhoL * uL**2 + pL,
        uL   * (EL + pL)
    ])

    FR = np.array([
        rhoR * uR,
        rhoR * uR**2 + pR,
        uR   * (ER + pR)
    ])

    # Compute HLL flux
    if SL > 0:
        return FL
    elif SL <= 0 and SR >= 0:
        return (SR * FL - SL * FR + SL * SR * (UR - UL)) / (SR - SL)
    else:
        return FR

def godunov(U, solver, dt, dx, N, **kwargs):
    num_vars = U.shape[1]
    flux = np.zeros((N + 1, num_vars))
    for i in range(N + 1):
        flux[i] = solver(U[i], U[i+1])
    dU = (dt / dx[:, np.newaxis]) * (flux[1:N+1] - flux[0:N])

    return dU

def MUSCL(U, solver, dt, dx, N, X, **kwargs):
    num_vars = U.shape[1]

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
    
    dU = (dt / dx[:, np.newaxis]) * (flux[1:N+1] - flux[0:N])

    return dU

def WENO(U, solver, dt, dx, N, X, **kwargs):
    num_vars = U.shape[1]
    epsilon = 1e-6

    d_L = np.array([1/10, 6/10, 3/10])

    def weno5_reconstruct_single_var(u_five_points):
    
        v0, v1, v2, v3, v4 = u_five_points
        d = d_L 

        q0 = (1/3) * v0 - (7/6) * v1 + (11/6) * v2
        q1 = -(1/6) * v1 + (5/6) * v2 + (1/3) * v3
        q2 = (1/3) * v2 + (5/6) * v3 - (1/6) * v4

        # Smoothness indicators beta_k.
        beta0 = (13/12) * (v0 - 2*v1 + v2)**2 + (1/4) * (v0 - 4*v1 + 3*v2)**2
        beta1 = (13/12) * (v1 - 2*v2 + v3)**2 + (1/4) * (v1 - v3)**2
        beta2 = (13/12) * (v2 - 2*v3 + v4)**2 + (1/4) * (3*v2 - 4*v3 + v4)**2
        betas = np.array([beta0, beta1, beta2])

        # Calculate alpha_k, which are intermediate weights.
        alphas = d / (epsilon + betas)**2
        
        # Calculate non-linear weights omega_k.
        omega_sum = np.sum(alphas)
        if omega_sum == 0:
            omegas = d # Fallback to linear weights if sum is zero
        else:
            omegas = alphas / omega_sum

        reconstructed_val = omegas[0] * q0 + omegas[1] * q1 + omegas[2] * q2
        return reconstructed_val

    UL_cell = np.zeros((N + 2, num_vars))
    UR_cell = np.zeros((N + 2, num_vars))

    for k in range(N + 2):

        for var_idx in range(num_vars):
            UL_cell[k, var_idx] = weno5_reconstruct_single_var(U[k:k+5, var_idx])

        for var_idx in range(num_vars):
            UR_cell[k, var_idx] = weno5_reconstruct_single_var(U[k:k+5, var_idx][::-1])

    flux = np.zeros((N + 1, num_vars))
    for i in range(N + 1):
        flux[i] = solver(UL_cell[i], UR_cell[i+1])

    dU = (dt / dx[:, np.newaxis]) * (flux[1:N+1] - flux[0:N])

    return dU

def calc_dt(grid_instance):
    """
    Calculates the maximum allowable time step (dt) for each refinement level
    in the grid, based on the CFL condition for cells at that level.

    Args:
        grid_instance (grid): An instance of your grid class.

    Returns:
        dict: A dictionary where keys are refinement levels (int) and values
              are the calculated maximum dt for cells at that level.
    """
    # Get all active cells, grouped by level
    cells_by_level = grid_instance.get_same_level_cells()

    dt_per_level = {}

    for level, cell_list in cells_by_level.items():

        prim_level = np.array([c.prim for c in cell_list])
        dx_level = np.array([c.dx for c in cell_list])

        p_level = prim_level[:, 2]
        rho_level = prim_level[:, 0]
        u_level = prim_level[:, 1]

        cs_level = np.sqrt(GAMMA * p_level / rho_level)
        c_level = np.abs(u_level) + cs_level

        # Prevent instability
        c_level = np.maximum(c_level, np.finfo(float).eps)

        dt_level = CFL * np.min(dx_level / c_level)        
        dt_per_level[level] = dt_level

    return dt_per_level

def method(order):
    if order == 1:
        print("Using Godunov Method")
        return godunov
    if order == 2:
        print("Using MUSCL")
        return MUSCL
    if order == 5:
        print("Using WENO")
        return WENO

############### Simulation ###############

def solve(solver, grid, t_final, order=1, **kwargs):

    t = 0
    calc_dU = method(order)
    history = []
    
    with tqdm(total=t_final, unit="s", desc="Solving Simulation") as pbar:
        while t < t_final:
            active_cells = grid.get_all_active_cells()
            N = len(active_cells)
            grid_prim = np.array([c.prim for c in active_cells])
            U = prim2con_grid(grid_prim)
            X = np.array([c.x for c in active_cells])
            dx = np.array([c.dx for c in active_cells])
            history.append(copy.deepcopy(grid))

            # Compute time step
            dt_dict = calc_dt(grid)
            dt = np.min(list(dt_dict.values()))

            if t + dt > t_final:
                dt = t_final - t

            # Compute fluxes at interfaces
            U_gc, X_gc = grid.generate_gc(U, X)

            # Update conserved variables
            dU = calc_dU(U=U_gc, solver=solver, dt=dt, dx=dx, N=N, X=X_gc)

            U -= dU
            grid.update(con2prim_grid(U))

            # Update time and add to history
            t += dt
            grid.t = t

            grid.flag_cells(**kwargs)
            grid.refine()
            grid.coarse()

            pbar.update(dt)

    print("FINISHED")
    
    return history

def new_solve(solver, grid, t_final, order=1, **kwargs):

    t = 0
    calc_dU = method(order)
    history = []

    with tqdm(total=t_final, unit="s", desc="Solving Simulation") as pbar:
        while t < t_final:
            history.append(copy.deepcopy(grid))

            grid.refine(id_only=False) # Refine all cell first
            active_cells = grid.get_all_active_cells()
            N = len(active_cells)
            grid_prim = np.array([c.prim for c in active_cells])
            X = np.array([c.x for c in active_cells])
            U = prim2con_grid(grid_prim)
            dx = np.array([c.dx for c in active_cells])

            # Compute time step
            dt_dict = calc_dt(grid)
            dt = np.min(list(dt_dict.values()))

            if t + dt > t_final:
                dt = t_final - t

            # Compute fluxes at interfaces
            U_gc, X_gc = grid.generate_gc(U, X)

            # Compute fluxes at interfaces
            dU = calc_dU(U=U_gc, solver=solver, dt=dt, dx=dx, N=N, X=X_gc)

            U -= dU
            grid.update(con2prim_grid(U))

            # Update time and add to history
            t += dt
            grid.t = t

            # To coarse or not to coarse
            def new_flag(epsilon, **kwargs):
                for c in range(0, N, 2):
                    cell_l = active_cells[c]
                    cell_r = active_cells[c+1]

                    avg = (cell_l.prim + cell_r.prim) / 2

                    diff_l = np.abs(cell_l.prim - avg)
                    diff_r = np.abs(cell_r.prim - avg)

                    #DEBUG_DIFF = np.concatenate([diff_l, diff_r])
                    #print(np.max(DEBUG_DIFF), np.min(DEBUG_DIFF), np.mean(DEBUG_DIFF))

                    if np.all(diff_l < epsilon) and np.all(diff_r < epsilon):
                        grid.coarsen_cell(active_cells[c].parent) # coarse all cell that has diff < epsilon
            
            new_flag(**kwargs)

            pbar.update(dt)

    print("FINISHED")
    
    return history