import numpy as np
from misc import *

class PPM_Interpolant:
    def __init__(self):
        self.NG = 4

    def update(self, prim, dx, bc_type):
        self.N = prim.shape[0]
        self.nvar = prim.shape[1]
        self.prim_gc = generate_gc(prim, self.NG, bc_type)
        self.dx_gc = generate_gc(dx, self.NG, bc_type)

        # Init variables
        self.a_j = np.zeros_like(self.prim_gc)
        self.a_jp1 = np.zeros_like(self.prim_gc)
        self.a_jp2 = np.zeros_like(self.prim_gc)
        self.a_jm1 = np.zeros_like(self.prim_gc)
        self.a_jm2 = np.zeros_like(self.prim_gc)

        self.xi_j = np.zeros(self.prim_gc.shape[0])
        self.xi_jm1 = np.zeros(self.prim_gc.shape[0])
        self.xi_jm2 = np.zeros(self.prim_gc.shape[0])
        self.xi_jp1 = np.zeros(self.prim_gc.shape[0])
        self.xi_jp2 = np.zeros(self.prim_gc.shape[0])

        self.aL = np.zeros_like(self.prim_gc)
        self.aR = np.zeros_like(self.prim_gc)

        # Update variables value
        self.lo = self.NG
        self.hi = self.NG + self.N - 1
        self.ib = self.lo - 2
        self.ie = self.hi + 1
        
        self.a_j[self.ib:self.ie+1] = self.prim_gc[self.ib:self.ie+1]
        self.a_jp1[self.ib:self.ie+1] = self.prim_gc[self.ib+1:self.ie+2]
        self.a_jp2[self.ib:self.ie+1] = self.prim_gc[self.ib+2:self.ie+3]
        self.a_jm1[self.ib:self.ie+1] = self.prim_gc[self.ib-1:self.ie]
        self.a_jm2[self.ib:self.ie+1] = self.prim_gc[self.ib-2:self.ie-1]

        self.xi_j[self.ib:self.ie+1] = self.dx_gc[self.ib:self.ie+1]
        self.xi_jp1[self.ib:self.ie+1] = self.dx_gc[self.ib+1:self.ie+2]
        self.xi_jp2[self.ib:self.ie+1] = self.dx_gc[self.ib+2:self.ie+3]
        self.xi_jm1[self.ib:self.ie+1] = self.dx_gc[self.ib-1:self.ie]
        self.xi_jm2[self.ib:self.ie+1] = self.dx_gc[self.ib-2:self.ie-1]

        self.xi_j = self.xi_j[:, np.newaxis]
        self.xi_jp2 = self.xi_jp2[:, np.newaxis]
        self.xi_jp1 = self.xi_jp1[:, np.newaxis]
        self.xi_jm1 = self.xi_jm1[:, np.newaxis]
        self.xi_jm2 = self.xi_jm2[:, np.newaxis]

        self.get_boundary(self.xi_jm1, self.xi_j, self.xi_jp1, self.xi_jp2, self.a_jm1, self.a_j, self.a_jp1, self. a_jp2)

    def get_mean_slope(self, xi_jm1, xi_j, xi_jp1, a_jm1, a_j, a_jp1, limit=True):
        factor1 = xi_j / (xi_jm1 + xi_j + xi_jp1 + np.finfo(float).eps)
        factor2 = (2 * xi_jm1 + xi_j) * (a_jp1 - a_j) / (xi_jp1 + xi_j + np.finfo(float).eps)
        factor3 = (xi_j + 2 * xi_jp1) * (a_j - a_jm1) / (xi_jm1 + xi_j + np.finfo(float).eps)

        daj = factor1 * (factor2 + factor3)

        if limit:
            con =  (a_jp1 - a_j) * (a_j - a_jm1) 
            daj[con >= 0] = (np.minimum(np.abs(daj), 2 * np.abs(a_j - a_jm1)) * np.sign(daj))[con >= 0]
            daj [con < 0] = 0

        return daj
    
    def get_boundary(self, xi_jm1, xi_j, xi_jp1, xi_jp2, a_jm1, a_j, a_jp1, a_jp2):
        
        da_j = self.get_mean_slope(xi_jm1, xi_j, xi_jp1, a_jm1, a_j, a_jp1)
        da_jp1 = self.get_mean_slope(xi_j, xi_jp1, xi_jp2, a_j, a_jp1, a_jp2)
        
        
        f1 = xi_j * (a_jp1 - a_j)
        f2 = 1 / (xi_jm1 + xi_j + xi_jp1 + xi_jp2 + np.finfo(float).eps)
        f3 = 2 * xi_jp1 * xi_j / (xi_j + xi_jp1 + np.finfo(float).eps)
        f4 = (xi_jm1 + xi_j) / (2 * xi_j + xi_jp1 + np.finfo(float).eps)
        f5 = (xi_jp2 + xi_jp1) / (2 * xi_jp1 + xi_j + np.finfo(float).eps)
        f6 = xi_j * da_jp1 * (xi_jm1 + xi_j) / (2 * xi_j + xi_jp1 + np.finfo(float).eps)
        f7 = xi_jp1 * da_j * (xi_jp1 + xi_jp2) / (xi_j + 2 * xi_jp1 + np.finfo(float).eps)

        a_boundary = self.a_j + f1 + f2 * (f3 * (f4 - f5) * (self.a_jp1 - self.a_j) - f6 + f7)
        return a_boundary
    
    def get_interface(self):
        # left of cell
        a_jmh = self.get_boundary(self.xi_jm2, self.xi_jm1, self.xi_j, self.xi_jp1, self.a_jm2, self.a_jm1, self.a_j, self.a_jp1)
        # right of cell
        a_jph = self.get_boundary(self.xi_jm1, self.xi_j, self.xi_jp1, self.xi_jp2, self.a_jm1, self.a_j, self.a_jp1, self.a_jp2)

        """
        self.aL = self.a_jm1 + 0.5 * da_jm1 
        self.aR = self.a_jp1 - 0.5 * da_jp1
        """
        
        # Limiting (Migone Eq. 45)
        a_jmh = np.minimum(a_jmh, np.maximum(self.a_j, self.a_jm1))
        a_jmh = np.maximum(a_jmh, np.minimum(self.a_j, self.a_jm1))

        a_jph = np.minimum(a_jph, np.maximum(self.a_j, self.a_jp1))
        a_jph = np.maximum(a_jph, np.minimum(self.a_j, self.a_jp1))

        dqf_m = self.a_j - a_jmh
        dqf_p = a_jph - self.a_j
        
        cond = dqf_m * dqf_p <= 0
        dqf_p[cond] = 0
        dqf_m[cond] = 0

        cond2 = np.abs(dqf_m) >= 2 * np.abs(dqf_p)
        dqf_m[~cond & cond2] = self.a_j[~cond & cond2] - 2 * dqf_p[~cond & cond2]

        cond3 = np.abs(dqf_p) >= 2 * np.abs(dqf_m)
        dqf_p[~cond & cond3] = self.a_j[~cond & cond3] + 2 * dqf_m[~cond & cond3]
        
        self.aR = a_jph
        self.aL = a_jmh
        self.a6 = 6.0 * self.a_j - 3.0 * (self.aR + self.aL)

    def reconstruct(self, cell_index, xi):
        if xi > 0.5 or xi < -0.5:
            raise ValueError("Xi Out of Bound.")
        
        self.get_interface()
        i = cell_index + self.NG

        return (self.aL + xi * (self.aR - self.aL + self.a6 * (1.0 - xi)))[i]

def PPM(U, solver, dt, dx, N, X, bc_type='outflow', **kwargs):
    ppm = PPM_Interpolant()
    q = con2prim_grid(U)

    ppm.update(q, dx, bc_type)
    ppm.get_interface()

    flux = np.zeros((N + 1, q.shape[1]))

    for i in range(N + 1):
        UL = prim2con(ppm.aR[i+ppm.NG-1]) # left of the edge
        UR = prim2con(ppm.aL[i+ppm.NG]) # right of the edge
        flux[i] = solver(UL, UR)
    
    dU = (1 / dx[:, np.newaxis]) * (flux[1:] - flux[:-1])

    return -dU