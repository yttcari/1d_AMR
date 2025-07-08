import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
from reconstruct import *
    
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

def dx_method(type, **kwargs):
    if type == 'godunov':
        return godunov(**kwargs)
    if type == 'MUSCL':
        return MUSCL(**kwargs)
    if type == 'WENO':
        return WENO(**kwargs)
    
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


############### Simulation ###############

def solve(solver, grid, t_final, dx_type='godunov', dt_type='rk4', **kwargs):

    t = 0
    print(f"Using {dx_type} in spatial and {dt_type} in temporal.")
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

            # Update conserved variables
            dU = dt_method(U=U, solver=solver, dt=dt, dx=dx, N=N, X=X, dt_type=dt_type, dx_type=dx_type)

            U += dU
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

def new_solve(solver, grid, t_final, dx_type='godunov', dt_type='rk4',**kwargs):

    t = 0
    print(f"Using {dx_type} in spatial and {dt_type} in temporal.")
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

            dU = dt_method(U=U, solver=solver, dt=dt, dx=dx, N=N, X=X, dt_type=dt_type, dx_type=dx_type)

            U += dU
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