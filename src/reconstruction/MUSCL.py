import numpy as np
from misc import generate_gc, prim2con_grid, con2prim_grid

class MUSCL_reconstruction:
    def __init__(self):
        self.NG = 2

    def update(self, prim, X, dx, bc_type):
        self.prim_gc = generate_gc(prim, self.NG, bc_type)
        self.dx_gc = generate_gc(dx, self.NG, bc_type)
        self.X_gc = self.generate_X_gc(X, self.NG)

        self.mp = self.get_mp()
        self.mm = self.get_mm()

        self.slope = self.mc_limiter(self.mp, self.mm)

        self.prim_L_gc, self.prim_R_gc = self.get_boudary()

        self.UL_gc = prim2con_grid(self.prim_L_gc)
        self.UR_gc = prim2con_grid(self.prim_R_gc)

    def mc_limiter(self, a, b):
        return np.where(a * b <= 0, 0.0, np.sign(a) * np.minimum(np.abs(a + b) / 2, np.minimum(2*np.abs(a), 2*np.abs(b))))

    def generate_X_gc(self, X, NG):
        N = len(X)
        X_with_gc = np.zeros(N + NG * 2)

        X_with_gc[NG:-NG] = X

        dx_left = X_with_gc[NG+1] - X_with_gc[NG]
        dx_right = X_with_gc[-NG-1] - X_with_gc[-NG-2]

        for i in range(NG):
            X_with_gc[NG - 1 - i] = X_with_gc[NG - i] - dx_left # Left ghost cells
            X_with_gc[NG + N + i] = X_with_gc[NG + N + i - 1] + dx_right # Right ghost cells

        return X_with_gc

    def get_mp(self):
        dX = self.X_gc[1:] - self.X_gc[:-1]
        # get slope from forward differencing
        slope_plus = np.zeros_like(self.prim_gc)
        slope_plus[:-1] = (self.prim_gc[1:] - self.prim_gc[:-1]) / dX[:, np.newaxis]

        return slope_plus

    def get_mm(self):
        dX = self.X_gc[1:] - self.X_gc[:-1]
        # get slope from backward differencing
        slope_minus = np.zeros_like(self.prim_gc)
        slope_minus[1:] = (self.prim_gc[1:] - self.prim_gc[:-1]) / dX[:, np.newaxis]

        return slope_minus
    
    def get_boudary(self):
        prim_L_gc = self.prim_gc - 0.5 * self.dx_gc[:, np.newaxis] * self.slope # left of cell i
        prim_R_gc = self.prim_gc + 0.5 * self.dx_gc[:, np.newaxis] * self.slope # right of cell i

        return prim_L_gc, prim_R_gc 

    def reconstruct(self, cell_index, xi):
        if xi > 0.5 or xi < -0.5:
            raise ValueError("Xi Out of Bound.")
        
        i = cell_index + self.NG

        slope = self.slope[i]
        dx = self.dx_gc[i]

        prim = self.prim_gc[i]

        return prim + xi * dx * slope


def MUSCL(U, solver, dt, dx, N, X, bc_type='outflow', **kwargs):
    num_vars = U.shape[1]
    prim = con2prim_grid(U)
    recon = MUSCL_reconstruction()
    recon.update(prim, X, dx, bc_type)
    
    NG = recon.NG

    UL_gc = recon.UL_gc
    UR_gc = recon.UR_gc

    flux = np.zeros((N + 1, num_vars))

    for i in range(N + 1):
        flux[i] = solver(UR_gc[i+NG-1], UL_gc[i+NG])
    
    dU = (1 / dx[:, np.newaxis]) * (flux[1:] - flux[:-1])

    return -dU