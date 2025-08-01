import numpy as np
import cell
from reconstruct import *
from misc import generate_gc

# Grid that stores box
class grid:
    def __init__(self, L, N):
        self.L = L
        self.N = N
        self.dx = {0: L/N}
        self.grid = {} 
        self.id_counter = 0
        self.reconstruction = ppm_refine
        self.t = 0

        self.bc_type = 'outflow'

        # Initialization
        for n in range(self.N):
            cell_id = self.get_next_id()
            new_cell = cell.cell(prim=np.zeros(3), xmin=n * self.dx[0], xmax=(n + 1) * self.dx[0],
                            id=cell_id)
            self.grid[cell_id] = new_cell 

        self.max_level = 5

    def update_max_level(self, new_max):
        self.max_level = new_max

    def update_reconstruction(self, method):
        if method == 'godunov':
            print("Using zero-th order reconstruction")
            self.reconstruction = zero_order
        elif method == 'MUSCL':
            print("Using linear reconstruction")
            self.reconstruction = linear
        elif method == 'PPM':
            print("Using PPM reconstruction")
            self.reconstruction = ppm_refine
        else:
            raise ValueError("The entered method is not implemented.")

    def __repr__(self):
        return(f"(L={self.L}, N={self.N}, dx={self.dx}, grid={self.grid}, id_counter={self.id_counter}")

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


    def refine_cell(self, parent_cell_id, active_cells):
        """
        Refines a specified parent cell into two children using linear interpolation.
        Uses minmod slope limiter for second-order accurate interpolation.
        """
        parent_cell = self.get_cell_by_id(parent_cell_id)
        if not parent_cell:
            raise ValueError(f"Error: Parent cell with ID {parent_cell_id} not found for refinement.")

        if not parent_cell.activating: # Already refined
            raise TypeError(f"Cell with ID {parent_cell_id} is already refined (not activating).")

        # Get neighboring cells for slope calculation
        child_level = parent_cell.level + 1
        child_left_id = self.get_next_id()
        child_right_id = self.get_next_id()

        child_left_prim, child_right_prim = self.reconstruction(active_cells, parent_cell)

        child_left = cell.cell(
            prim=child_left_prim,
            xmin=parent_cell.xmin, xmax=parent_cell.x,
            id=child_left_id, level=child_level, parent=parent_cell_id 
        )
        child_right = cell.cell(
            prim=child_right_prim,
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
    
    def flag_cells(self, refine_epsilon=0.5, buffer_layers=1, corase_epsilon=0.3, **kwargs):
        active_cell = self.get_all_active_cells()
        N = len(active_cell)

        prim = np.array([c.prim for c in active_cell])

        prim_with_gc = np.zeros((prim.shape[0] + 2, prim.shape[1]))

        prim_with_gc = generate_gc(prim, 1, self.bc_type)#np.pad(prim, ((1, 1), (0, 0)), 'edge')

        # Central difference
        dU = prim_with_gc[2:, :] - prim_with_gc[:-2, :]

        grad = np.abs(dU)
        refine_cell_index = np.unique(np.where(grad > refine_epsilon)[0])
        coarse_cell_index = np.unique(np.where(grad < corase_epsilon)[0])

        # find cell to be flagged
        for i in refine_cell_index:
            if active_cell[i].level < self.max_level:
                active_cell[i].need_refine = True

            for buffer_no in range(1, buffer_layers+1):
                cell_id_bef = np.min([i + buffer_no, N-1])
                cell_id_after = np.max([i - buffer_no, 0])

                cell_bef = active_cell[cell_id_bef]
                cell_after = active_cell[cell_id_after]

                if cell_bef.level < self.max_level:
                    cell_bef.need_refine = True

                if cell_after.level < self.max_level:
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
                self.refine_cell(c.id, active_cell)
                c.need_refine = False
            if not id_only and c.level < self.max_level:
                self.refine_cell(c.id, active_cell)

    def coarse(self, **kwargs):
        active_cell = self.get_all_active_cells(**kwargs)

        for c in active_cell:
            if c.need_coarse and c.id in self.grid:
                self.coarsen_cell(c.parent)
                c.need_coarse = False


# ===== Refinement Method within grid =====
from reconstruction import MUSCL
# TODO Re-use reconstruction function?
def zero_order(active_cells, parent_cell):
    child_left_prim =parent_cell.prim
    child_right_prim = parent_cell.prim
    return child_left_prim, child_right_prim

def linear(active_cells, parent_cell):
    parent_index = active_cells.index(parent_cell)

    left = active_cells[parent_index - 1] if parent_index > 0 else None
    right = active_cells[parent_index + 1] if parent_index < len(active_cells) - 1 else None

    if left is not None and right is not None:
        # Use 3-point stencil for slope calculation
        U_arr = np.array([left.prim, parent_cell.prim, right.prim])
        X_arr = np.array([left.x, parent_cell.x, right.x])
            
        # Calculate slope using minmod limiters)
        sigma = MUSCL.get_slope(U_arr, X_arr)
        # Use slope at center point (index 1)
        slope = sigma[1]

    elif left is not None:
            # Only left neighbor available - use one-sided difference
        slope = (np.array(parent_cell.prim) - np.array(left.prim)) / (parent_cell.x - left.x)
            
    elif right is not None:
            # Only right neighbor available - use one-sided difference  
        slope = ((np.array(right.prim) - np.array(parent_cell.prim)) / (right.x - parent_cell.x))
            
    else:
            # No neighbors available - use zero slope (constant interpolation)
        slope = np.zeros_like(parent_cell.prim)

    # Calculate child cell centers
    child_left_center = (parent_cell.xmin + parent_cell.x) / 2
    child_right_center = (parent_cell.x + parent_cell.xmax) / 2
        
    # Linear interpolation: U(x) = U_parent + slope * (x - x_parent)
    child_left_prim = parent_cell.prim + slope * (child_left_center - parent_cell.x)
    child_right_prim = parent_cell.prim + slope * (child_right_center - parent_cell.x)

    return child_left_prim, child_right_prim

def ppm_refine(active_cells, parent_cell):
    parent_index = active_cells.index(parent_cell)
    
    # 4-point stencil
    if parent_index == 0:
        left_prim = np.array(parent_cell.prim)
    else:
        left_prim = np.array(active_cells[parent_index - 1].prim)
    
    parent_prim = np.array(parent_cell.prim)
    
    if parent_index >= len(active_cells) - 1:
        right_prim = np.array(parent_cell.prim)
    else:
        right_prim = np.array(active_cells[parent_index + 1].prim)
    
    if parent_index >= len(active_cells) - 2:
        right_right_prim = np.array(right_prim)
    else:
        right_right_prim = np.array(active_cells[parent_index + 2].prim)
    
    a_data = np.vstack((left_prim, parent_prim, right_prim, right_right_prim))
    
    # da0 = 0.5 * (a[i+1] - a[i-1]) for cell i 
    # dap = 0.5 * (a[i+2] - a[i]) for cell i
    da0 = 0.5 * (a_data[2] - a_data[0]) 
    dap = 0.5 * (a_data[3] - a_data[1])
    
    # van Leer limiting
    dl = a_data[1] - a_data[0] 
    dr = a_data[2] - a_data[1]
    
    da0 = np.where(dl * dr < 0, 0.0, 
                   np.sign(da0) * np.minimum(np.abs(da0),
                                           2.0 * np.minimum(np.abs(dl), np.abs(dr))))
    
    dl = a_data[2] - a_data[1] 
    dr = a_data[3] - a_data[2] 
    
    dap = np.where(dl * dr < 0, 0.0,
                   np.sign(dap) * np.minimum(np.abs(dap),
                                           2.0 * np.minimum(np.abs(dl), np.abs(dr))))
    
    aint = 0.5 * (a_data[1] + a_data[2]) - (1.0 / 6.0) * (dap - da0)
    ap = aint.copy()
    
    if parent_index > 0:
        if parent_index == 1:
            left_left = np.array(active_cells[0].prim)
        else:
            left_left = np.array(active_cells[parent_index - 2].prim) if parent_index >= 2 else left_prim
        
        left_stencil = np.vstack((left_left, left_prim, parent_prim, right_prim))
        
        # left neighbor
        da0_left = 0.5 * (left_stencil[2] - left_stencil[0])
        dap_left = 0.5 * (left_stencil[3] - left_stencil[1])
        
        # Limit slopes
        dl_left = left_stencil[1] - left_stencil[0]
        dr_left = left_stencil[2] - left_stencil[1]
        da0_left = np.where(dl_left * dr_left < 0, 0.0,
                           np.sign(da0_left) * np.minimum(np.abs(da0_left),
                                                         2.0 * np.minimum(np.abs(dl_left), np.abs(dr_left))))
        
        dl_left = left_stencil[2] - left_stencil[1]
        dr_left = left_stencil[3] - left_stencil[2]
        dap_left = np.where(dl_left * dr_left < 0, 0.0,
                           np.sign(dap_left) * np.minimum(np.abs(dap_left),
                                                         2.0 * np.minimum(np.abs(dl_left), np.abs(dr_left))))
        
        aint_left = 0.5 * (left_stencil[1] + left_stencil[2]) - (1.0 / 6.0) * (dap_left - da0_left)
        am = aint_left.copy()
    else:
        am = a_data[1].copy()
    
    # Monotonicity
    a = a_data[1]
    test = (ap - a) * (a - am) < 0

    am = np.where(test, a, am)
    ap = np.where(test, a, ap)
    
    da = ap - am
    
    testm = da * (a - 0.5 * (am + ap)) > da**2 / 6
    am = np.where(testm, 3.0*a - 2.0*ap, am)
    
    testp = -da**2 / 6 > da * (a - 0.5 * (am + ap))
    ap = np.where(testp, 3.0*a - 2.0*am, ap)
    
    a6 = 6.0 * a - 3.0 * (am + ap)
    
    dx = parent_cell.xmax - parent_cell.xmin
    

    child_left_center = parent_cell.xmin + 0.25 * dx  
    child_right_center = parent_cell.xmin + 0.75 * dx 
    
    xi_L = (child_left_center - parent_cell.x) / dx  
    xi_R = (child_right_center - parent_cell.x) / dx
    
    # parabola = am + x(ap - am + a6(1 - x))
    def evaluate_parabola(xi, am, ap, a6):
        return am + xi * (ap - am + a6 * (1.0 - xi))
    
    xi_L_edge = xi_L + 0.5  
    xi_R_edge = xi_R + 0.5 
    
    child_left_prim = evaluate_parabola(xi_L_edge, am, ap, a6)
    child_right_prim = evaluate_parabola(xi_R_edge, am, ap, a6)
    
    return child_left_prim, child_right_prim