import numpy as np
import cell

# Grid that stores box
class grid:
    def __init__(self, L, N):
        self.L = L
        self.N = N
        self.dx = {0: L/N}
        self.grid = {} 
        self.id_counter = 0
        self.t = 0

        # Initialization
        for n in range(self.N):
            cell_id = self.get_next_id()
            new_cell = cell.cell(prim=np.zeros(3), xmin=n * self.dx[0], xmax=(n + 1) * self.dx[0],
                            id=cell_id)
            self.grid[cell_id] = new_cell 

        self.max_level = 3

    def update_max_level(self, new_max):
        self.max_level = new_max

    def __repr__(self):
        return(f"(L={self.L}, N={self.N}, dx={self.dx}, grid={self.grid}, id_counter={self.id_counter}, t={self.t})")

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
        
        parent_cell = self.get_cell_by_id(parent_cell_id)
        if not parent_cell:
            raise ValueError(f"Error: Parent cell with ID {parent_cell_id} not found for coarsening.")

        if parent_cell.activating or not parent_cell.children:
            raise TypeError(f"Cell with ID {parent_cell_id} is already activating or has no children to coarsen.")

        # Update parent prim
        parent_cell.prim = np.zeros(np.array(parent_cell.prim).shape)

        for child_id in parent_cell.children:
            child = self.get_cell_by_id(child_id)

            parent_cell.prim += child.prim

        parent_cell.prim /= len(parent_cell.children)

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

    def get_same_level_cells(self):
        """
        Retrieves all active cells, grouped by their refinement level.
        Returns a dictionary where keys are levels (int) and values are lists of cell objects
        at that level, sorted by their xmin.
        """
        all_active_cells = self.get_all_active_cells()

        cells_by_level = {}

        for cell_obj in all_active_cells:
            level = cell_obj.level
            if level not in cells_by_level:
                cells_by_level[level] = []
            cells_by_level[level].append(cell_obj)

        # sort cell order
        for level in cells_by_level:
            cells_by_level[level].sort(key=lambda c: c.xmin)

        return cells_by_level
    
    def flag_cells(self, refine_epsilon=0.5, buffer_layers=1, max_level=3, corase_epsilon=0.3, **kwargs):
        active_cell = self.get_all_active_cells()
        N = len(active_cell)

        prim = np.array([c.prim for c in active_cell])
        X = np.array([c.x for c in active_cell])

        prim_with_gc = np.zeros((prim.shape[0] + 2, prim.shape[1]))
        X_with_gc = np.zeros(N + 2)

        # Make ghost cell
        prim_with_gc[1:-1] = prim
        X_with_gc[1:-1] = X

        prim_with_gc[0, :] = prim_with_gc[1, :]
        prim_with_gc[-1, :] = prim_with_gc[-2, :]

        X_with_gc[0] = 0 - X_with_gc[1]
        X_with_gc[-1] = self.L + (self.L - X_with_gc[-2])

        # Central difference
        dU = prim_with_gc[2:, :] - prim_with_gc[:-2, :]
        #dx = X_with_gc[2:] - X_with_gc[:-2]

        grad = np.abs(dU)
        #print(f"Average Gradient: {np.round(np.average(grad), 2)}, Max: {np.round(np.max(grad), 2)}, Min: {np.round(np.min(grad), 2)}")
        refine_cell_index = np.unique(np.where(grad > refine_epsilon)[0])
        coarse_cell_index = np.unique(np.where(grad < corase_epsilon)[0])

        # find cell to be flagged
        for i in refine_cell_index:
            if active_cell[i].level < max_level:
                active_cell[i].need_refine = True

            for buffer_no in range(1, buffer_layers+1):
                cell_id_bef = np.min([i + buffer_no, N-1])
                cell_id_after = np.max([i - buffer_no, 0])

                cell_bef = active_cell[cell_id_bef]
                cell_after = active_cell[cell_id_after]

                if cell_bef.level < max_level:
                    cell_bef.need_refine = True

                if cell_after.level < max_level:
                    cell_after.need_refine = True

        for i in coarse_cell_index:
            if active_cell[i].level > 0:
                active_cell[i].need_coarse = True

            for buffer_no in range(1, buffer_layers+1):
                cell_id_bef = np.min([i + buffer_no, N-1])
                cell_id_after = np.max([i - buffer_no, 0])

                if active_cell[cell_id_bef].level > 0:
                    active_cell[cell_id_bef].need_coarse = True

                if active_cell[cell_id_after].level > 0:
                    active_cell[cell_id_after].need_coarse = True


    def refine(self, id_only=True, **kwargs):
        active_cell = self.get_all_active_cells(**kwargs)

        for c in active_cell:
            if c.need_refine and id_only:
                self.refine_cell(c.id)
                c.need_refine = False
            if not id_only and c.level < self.max_level:
                self.refine_cell(c.id)

    def coarse(self, **kwargs):
        active_cell = self.get_all_active_cells(**kwargs)

        for c in active_cell:
            if c.need_coarse and c.id in self.grid:
                self.coarsen_cell(c.parent)
                c.need_coarse = False