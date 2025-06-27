import numpy as np
import matplotlib.pyplot as plt

CFL = 0.5
GAMMA = 1.4

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

def con2prim_grid(con):
    tmp = []
    for i in con:
        tmp.append(con2prim(i))

    return np.array(tmp)

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
        rho_level = np.maximum(rho_level, np.finfo(float).eps)
        p_level = np.maximum(p_level, np.finfo(float).eps)

        cs_level = np.sqrt(GAMMA * p_level / rho_level)
        c_level = np.abs(u_level) + cs_level

        # Prevent instability
        c_level = np.maximum(c_level, np.finfo(float).eps)

        dt_level = CFL * np.min(dx_level / c_level)        
        dt_per_level[level] = dt_level

    return dt_per_level

def solve(solver, grid, t_final):
    active_cells = grid.get_all_active_cells()
    N = len(active_cells)

    grid_prim = np.array([c.prim for c in active_cells])
    U = prim2con_grid(grid_prim)
    dx = np.array([c.dx for c in active_cells])
    X = np.array([c.x for c in active_cells])
    t = 0

    history = []

    while t < t_final:
        history.append(np.concatenate((X.reshape(-1, 1), U), axis=1))

        print(f"Timestep: {t} s")
        # Compute time step
        
        # dt   = calc_dt(grid)
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

        # Update time and add to history
        t += dt

    print("FINISHED")
    grid.update(U)
    
    return np.array(history)