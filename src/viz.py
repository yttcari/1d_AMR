import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot_amr_grid(grid_instance, title="AMR Grid Structure", label=False):
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
    plt.show()

def plot_amr_value(grid, ax=None):
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

    plt.tight_layout()

def animate(history, filename=None, fps=10, dpi=100):
    """
    Creates and displays an animation of the AMR solution using the collected history data.

    Args:
        history (list): A list of dictionaries, where each dictionary contains
                        {'time', 'active_prims', 'active_x_coords'} for a snapshot.
        fps (int): Frames per second for the animation.
    """

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("AMR Simulation Results")

    all_densities = np.concatenate([h[:,0] for h in history])
    all_speeds = np.concatenate([h[:,1] for h in history])
    all_pressures = np.concatenate([h[:,2] for h in history])

    ax[0].set_ylim(all_densities.min() * 0.95, all_densities.max() * 1.05)
    ax[1].set_ylim(all_speeds.min() * 0.95 - 0.1, all_speeds.max() * 1.05 + 0.1)
    ax[2].set_ylim(all_pressures.min() * 0.95, all_pressures.max() * 1.05)

    x_min_global = 0.0
    x_max_global = 1.0
    for subplot_ax in ax:
        subplot_ax.set_xlim(x_min_global, x_max_global)


    def update(frame_idx):
        """
        Update function for FuncAnimation. This function is called for each frame
        and updates the plot with the data for that frame_idx.
        """
        snapshot = history[frame_idx]
        X_current = snapshot[:, 0]
        prim_current = snapshot[:, 1:]

        ax[0].cla()
        ax[1].cla()
        ax[2].cla()

        ax[0].set_xlabel(r"$x$")
        ax[0].set_ylabel(r"$\rho$")
        ax[0].plot(X_current, prim_current[:, 0], 'b-')
        ax[0].set_xlim(x_min_global, x_max_global) 
        ax[0].set_ylim(all_densities.min() * 0.95, all_densities.max() * 1.05)


        ax[1].set_xlabel(r"$x$")
        ax[1].set_ylabel(r"$u$")
        ax[1].set_title("Speed")
        ax[1].plot(X_current, prim_current[:, 1], 'g-')
        ax[1].set_xlim(x_min_global, x_max_global)
        ax[1].set_ylim(all_speeds.min() * 0.95 - 0.1, all_speeds.max() * 1.05 + 0.1)


        ax[2].set_xlabel(r"$x$")
        ax[2].set_ylabel(r"$P$")
        ax[2].set_title("Pressure")
        ax[2].plot(X_current, prim_current[:, 2], 'r-')
        ax[2].set_xlim(x_min_global, x_max_global)
        ax[2].set_ylim(all_pressures.min() * 0.95, all_pressures.max() * 1.05)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    print(f"Preparing animation with {len(history)} frames for display...")
    ani = FuncAnimation(fig, update, frames=len(history), interval=1000/fps, blit=False)
    if filename is not None:
        ani.save(filename, writer='pillow', fps=fps, dpi=dpi)