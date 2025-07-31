import numpy as np
from FVM import GAMMA

def sod_rod_tube(grid):
    LEFT = np.array([1.0, 0.0, 1.0])
    RIGHT = np.array([0.125, 0.0, 0.1])

    x_diaphragm = grid.L / 2

    #  [rho_L, u_L, P_L, rho_R, u_R, P_R, gamma, x_diaphragm] for plotting analytical solution
    init_con = np.concatenate([LEFT, RIGHT, np.array([GAMMA, x_diaphragm])])

    active_cells = grid.get_all_active_cells()

    if not active_cells:
        print("No active cells found to initialize.")
        return

    cell_x_coords = np.array([cell.x for cell in active_cells])

    # Iterate through active cells and set their primitive variables directly
    for i, cell in enumerate(active_cells):
        if cell_x_coords[i] < x_diaphragm:
            cell.prim = list(LEFT)
        else:
            cell.prim = list(RIGHT)

    return grid, init_con

def lax_tube(grid):
    LEFT = np.array([0.445, 0.698, 3.528])
    RIGHT = np.array([0.5, 0.0, 0.576])

    x_diaphragm = grid.L / 2

    #  [rho_L, u_L, P_L, rho_R, u_R, P_R, gamma, x_diaphragm] for plotting analytical solution
    #init_con = np.concatenate([LEFT, RIGHT, np.array([GAMMA, x_diaphragm])])

    active_cells = grid.get_all_active_cells()

    if not active_cells:
        print("No active cells found to initialize.")
        return

    cell_x_coords = np.array([cell.x for cell in active_cells])

    # Iterate through active cells and set their primitive variables directly
    for i, cell in enumerate(active_cells):
        if cell_x_coords[i] < x_diaphragm:
            cell.prim = list(LEFT)
        else:
            cell.prim = list(RIGHT)

    return grid