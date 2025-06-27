import numpy as np
import matplotlib.pyplot as plt

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

def plot_amr_value(grid):
    active_cells = grid.get_all_active_cells()

    prim = np.array([cell.prim for cell in active_cells])
    X = np.array([cell.x for cell in active_cells])

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