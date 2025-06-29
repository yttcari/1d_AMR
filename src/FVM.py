import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
    
CFL = 0.5
GAMMA = 1.4

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
        if internal_energy < 0: internal_energy = 1e-10

        pressure = internal_energy * (GAMMA - 1)
        
        # MODIFIED LINE: Assign a NumPy array directly to the row
        prim[i] = np.array([rho, u_vel, pressure], dtype=float) 
        
    return prim

def prim2con_grid(con):
    tmp = []
    for i in con:
        tmp.append(prim2con(i))

    return np.array(tmp)


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

""" SAME dx ONLY
def calc_dt(grid):
    active_cells = grid.get_all_active_cells()

    prim = np.array([c.prim for c in active_cells])
    dx = np.array(list(grid.dx.values()))

    p = prim[:, 2]
    rho = prim[:, 0]
    u = prim[:, 1]

    cs   = np.sqrt(GAMMA * p / rho)
    c = np.min(np.abs(u) + cs)

    dt = CFL * (dx / c)
    return dt
"""

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

        # Prevent instability
        rho_level = np.max(rho_level, np.finfo(float).eps)
        p_level = np.max(p_level, np.finfo(float).eps)

        cs_level = np.sqrt(GAMMA * p_level / rho_level)
        c_level = np.abs(u_level) + cs_level

        # Prevent instability
        c_level = np.maximum(c_level, np.finfo(float).eps)

        dt_level = CFL * np.min(dx_level / c_level)        
        dt_per_level[level] = dt_level

    return dt_per_level

def solve(solver, grid, t_final, **kwargs):

    t = 0

    history = []

    with tqdm(total=t_final, unit="s", desc="Solving Simulation") as pbar:
        while t < t_final:
            active_cells = grid.get_all_active_cells()
            N = len(active_cells)
            grid_prim = np.array([c.prim for c in active_cells])
            U = prim2con_grid(grid_prim)
            dx = np.array([c.dx for c in active_cells])
            history.append(copy.deepcopy(grid))

            # Compute time step
            dt_dict = calc_dt(grid)
            dt = np.min(list(dt_dict.values()))

            if t + dt > t_final:
                dt = t_final - t

            # Compute fluxes at interfaces
            flux = np.zeros((N+1, 3))
            for i in range(1,N):
                flux[i] = solver(U[i-1], U[i])

            # Boundary conditions on flux
            flux[ 0] = flux[ 1]
            flux[-1] = flux[-2]

            # Update conserved variables
            dU = (dt / dx)[:, np.newaxis] * (flux[1:] - flux[:-1])

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

def new_solve(solver, grid, t_final, **kwargs):

    t = 0

    history = []

    with tqdm(total=t_final, unit="s", desc="Solving Simulation") as pbar:
        while t < t_final:
            history.append(copy.deepcopy(grid))

            grid.refine(id_only=False) # Refine all cell first
            active_cells = grid.get_all_active_cells()
            N = len(active_cells)
            grid_prim = np.array([c.prim for c in active_cells])
            U = prim2con_grid(grid_prim)
            dx = np.array([c.dx for c in active_cells])

            # Compute time step
            dt_dict = calc_dt(grid)
            dt = np.min(list(dt_dict.values()))

            if t + dt > t_final:
                dt = t_final - t

            # Compute fluxes at interfaces
            flux = np.zeros((N+1, 3))
            for i in range(1,N):
                flux[i] = solver(U[i-1], U[i])

            # Boundary conditions on flux
            flux[ 0] = flux[ 1]
            flux[-1] = flux[-2]

            # Update conserved variables
            dU = (dt / dx)[:, np.newaxis] * (flux[1:] - flux[:-1])

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