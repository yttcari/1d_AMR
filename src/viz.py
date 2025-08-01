import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
import analytical

def plot_analytical(X, t, type, init_con, ax):
    # ax: list of ax from subplots
    if type == 'sod':
        rho_ana, u_ana, P_ana = analytical.get_sod_solution(X, t, *init_con)

        ax[0].plot(X, rho_ana, '--')
        ax[1].plot(X, u_ana, '--')
        ax[2].plot(X, P_ana, '--')
    elif type == 'plane':
        rho_ana, u_ana, P_ana = analytical.get_plane_wave_solution(X, t, *init_con)

        ax[0].plot(X, rho_ana, '--')
        ax[1].plot(X, u_ana, '--')
        ax[2].plot(X, P_ana, '--')

def plot_amr_grid(grid_instance, title="AMR Grid Structure", label=False, ax=None):
    """
    Plots the 1D AMR grid hierarchy on a number line.
    Visualizes 'activating' cells (leaf nodes) using their level for vertical offset.
    """
    active_cells = grid_instance.get_all_active_cells()

    if not active_cells:
        print("No active cells to plot.")
        return

    min_x = min(cell_obj.xmin for cell_obj in active_cells)
    max_x = max(cell_obj.xmax for cell_obj in active_cells)

    # Max level might be 0 if no refinement, so handle it
    max_level = max(cell_obj.level for cell_obj in active_cells) if active_cells else 0
    y_offset_multiplier = 0.05 * (max_x - min_x) # Adjust for desired vertical spacing

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    plotted_levels = set() # To ensure unique labels in legend

    for cell_obj in active_cells:
        x_start = cell_obj.xmin
        x_end = cell_obj.xmax
        level = cell_obj.level
        cell_id = cell_obj.id

        y_pos = level * y_offset_multiplier

        # Add label only once per level for the legend
        label_text = None
        if level not in plotted_levels:
            label_text = f'Level {level}'
            plotted_levels.add(level)

        ax.plot([x_start, x_end], [y_pos, y_pos],
                linewidth=2 + level*0.5, # Thicker lines for finer levels
                color='blue',
                marker='|', markersize=8, markeredgewidth=1, # Show cell boundaries
                label=label_text)

        if label:
            # Add cell ID and level as text on the plot
            ax.text(cell_obj.x, y_pos + y_offset_multiplier * 0.1,
                f"ID:{cell_id}\nL:{level}",
                fontsize=8, ha='center', va='bottom', color='black')

    ax.set_title(title)
    ax.set_xlabel("X-coordinate")
    ax.set_yticks([]) # Hide y-axis ticks as it's a conceptual "number line"
    ax.grid(axis='x', linestyle='--', alpha=0.7) # Grid lines only on x-axis

    # Set x-axis limits slightly beyond the min/max x of cells for padding
    ax.set_xlim(min_x - 0.05 * (max_x - min_x), max_x + 0.05 * (max_x - min_x))

    # Show legend only if there are multiple levels
    if len(plotted_levels) > 1:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    if ax is None:
        plt.show()

def plot_amr_value(grid, ax=None, type=None, init_con=None):
    active_cells = grid.get_all_active_cells()

    prim = np.array([cell.prim for cell in active_cells])
    X = np.array([cell.x for cell in active_cells])

    if ax is None:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].set_xlabel(r"$x$")
    ax[0].set_ylabel(r"$\rho$")
    ax[0].set_title("Density")

    ax[1].set_xlabel(r"$x$")
    ax[1].set_ylabel(r"$u$")
    ax[1].set_title("Speed")

    ax[2].set_xlabel(r"$x$")
    ax[2].set_ylabel(r"$P$")
    ax[2].set_title("Pressure")

    for i in range(len(ax)):
        ax[i].plot(X, prim[:, i])

    if type is not None:
        plot_analytical(X, grid.t, type, init_con, ax)

    plt.tight_layout()

def get_prim_history(grid_history):
    T = len(grid_history)

    rho, u, p, X = [], [], [], []

    for t in range(T):
        grid = grid_history[t]
        active_cell = grid.get_all_active_cells()

        prim = np.array([c.prim for c in active_cell])
        rho.append(prim[:, 0])
        u.append(prim[:, 1])
        p.append(prim[:, 2])
        X.append([c.x for c in active_cell])

    return rho, u, p, X

def animate(history, filename=None, fps=10, dpi=100, type=None, init_con=None):
    """
    Creates and displays an animation of the AMR solution using the collected history data.

    Args:
        history (list): A list of dictionaries, where each dictionary contains
                        {'time', 'active_prims', 'active_x_coords'} for a snapshot.
        fps (int): Frames per second for the animation.
    """

    fig, ax = plt.subplots(1, 4, figsize=(15, 5))

    all_densities, all_speeds, all_pressures, all_X = get_prim_history(history)

    rho_ymin, rho_ymax = np.min(np.concatenate(all_densities)), np.max(np.concatenate(all_densities))
    u_ymin, u_ymax = np.min(np.concatenate(all_speeds)), np.max(np.concatenate(all_speeds))
    p_ymin, p_ymax = np.min(np.concatenate(all_pressures)), np.max(np.concatenate(all_pressures))

    print(rho_ymax, rho_ymin)
    ax[0].set_ylim(rho_ymin, rho_ymax)
    ax[1].set_ylim(u_ymin, u_ymax)
    ax[2].set_ylim(p_ymin, p_ymax)


    x_min_global = 0.0
    x_max_global = 1.0
    for subplot_ax in ax:
        subplot_ax.set_xlim(x_min_global, x_max_global)


    def update(frame_idx):
        """
        Update function for FuncAnimation. This function is called for each frame
        and updates the plot with the data for that frame_idx.
        """

        ax[0].cla()
        ax[1].cla()
        ax[2].cla()
        ax[3].cla()

        if type is not None:
            plot_analytical(all_X[frame_idx], history[frame_idx].t, type, init_con, ax)

        ax[0].set_xlabel(r"$x$")
        ax[0].set_ylabel(r"$\rho$")
        ax[0].plot(all_X[frame_idx], all_densities[frame_idx], 'b-')
        ax[0].set_xlim(x_min_global, x_max_global) 
        ax[0].set_ylim(rho_ymin, rho_ymax)

        ax[1].set_xlabel(r"$x$")
        ax[1].set_ylabel(r"$u$")
        ax[1].set_title("Speed")
        ax[1].plot(all_X[frame_idx], all_speeds[frame_idx], 'g-')
        ax[1].set_xlim(x_min_global, x_max_global)
        ax[1].set_ylim(u_ymin, u_ymax)


        ax[2].set_xlabel(r"$x$")
        ax[2].set_ylabel(r"$P$")
        ax[2].set_title("Pressure")
        ax[2].plot(all_X[frame_idx], all_pressures[frame_idx], 'r-')
        ax[2].set_xlim(x_min_global, x_max_global)
        ax[2].set_ylim(p_ymin, p_ymax)

        plot_amr_grid(history[frame_idx], ax=ax[3])

        fig.suptitle(f"AMR Simulation Results at t={np.round(history[frame_idx].t, 2)} s")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    print(f"Preparing animation with {len(history)} frames for display...")
    ani = FuncAnimation(fig, update, frames=tqdm(range(len(history))), interval=1000/fps, blit=False)
    if filename is not None:
        ani.save(filename, writer='pillow', fps=fps, dpi=dpi)