import numpy as np
import cell

# Grid that stores box
class grid:
    def __init__(self, L, N):
        self.L = L
        self.N = N
        self.dx0 = L / N
        self.grid = {} 
        self.id_counter = 0

        # Initialization
        for n in range(self.N):
            cell_id = self.get_next_id()
            new_cell = cell.cell(prim=np.zeros(3), xmin=n * self.dx0, xmax=(n + 1) * self.dx0,
                            id=cell_id)
            self.grid[cell_id] = new_cell 

    def get_next_id(self):
        """Generates a unique ID for a new cell."""
        new_id_val = self.id_counter
        self.id_counter += 1
        return new_id_val

    def get_cell_by_id(self, cell_id):
        """Retrieves a cell object from the central dictionary by its ID (O(1) lookup)."""
        return self.grid.get(cell_id)

    def get_all_active_cells(self):
        """
        Retrieves all 'activating' (active/leaf) cells from the entire grid hierarchy.
        This traverses the tree using BFS and looks up child objects by ID.
        """
        active_cell = []

        for single_cell in self.grid.values():
            if single_cell.activating:
                active_cell.append(single_cell)

        active_cell.sort(key=lambda x: x.xmin)
        return active_cell


    def refine_cell(self, parent_cell_id):
        """
        Refines a specified parent cell into two children.
        Manages ID assignment, updates cell links (parent/children), and updates the grid dictionary.
        """
        parent_cell = self.get_cell_by_id(parent_cell_id)
        if not parent_cell:
            raise ValueError(f"Error: Parent cell with ID {parent_cell_id} not found for refinement.")

        if not parent_cell.activating: # Already refined
            raise TypeError(f"Cell with ID {parent_cell_id} is already refined (not activating).")

        # Create two new child cells, which inherit primitive variables
        child_level = parent_cell.level + 1
        child_left_id = self.get_next_id()
        child_right_id = self.get_next_id()

        child_left = cell.cell(
            prim=parent_cell.prim,
            xmin=parent_cell.xmin, xmax=parent_cell.x,
            id=child_left_id, level=child_level, parent=parent_cell_id 
        )
        child_right = cell.cell(
            prim=parent_cell.prim,
            xmin=parent_cell.x, xmax=parent_cell.xmax,
            id=child_right_id, level=child_level, parent=parent_cell_id
        )

        parent_cell.children.append(child_left_id)
        parent_cell.children.append(child_right_id)

        parent_cell.activating = False

        # Add new children to the central grid dictionary
        self.grid[child_left_id] = child_left
        self.grid[child_right_id] = child_right

        return child_left_id, child_right_id

    def coarsen_cell(self, parent_cell_id):
        """
        Coarsens a specified parent cell by removing its children from the hierarchy.
        Removes children (and their sub-children) from the central grid dictionary.
        """
        parent_cell = self.get_cell_by_id(parent_cell_id)
        if not parent_cell:
            raise ValueError(f"Error: Parent cell with ID {parent_cell_id} not found for coarsening.")

        if parent_cell.activating or not parent_cell.children:
            raise TypeError(f"Cell with ID {parent_cell_id} is already activating or has no children to coarsen.")

        # Recursively remove children
        def _remove_subtree_from_grid(cell_id_to_remove):
            cell_obj = self.get_cell_by_id(cell_id_to_remove)
            if cell_obj:
                for child_id in list(cell_obj.children):
                    _remove_subtree_from_grid(child_id)
                if cell_id_to_remove in self.grid: 
                    del self.grid[cell_id_to_remove]

        for child_id in list(parent_cell.children):
            _remove_subtree_from_grid(child_id)

        parent_cell.children = [] 
        parent_cell.activating = True

    def update(self, prim):
        active_cells = self.get_all_active_cells()

        for i, cell in enumerate(active_cells):
            cell.update(prim[i])