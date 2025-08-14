from misc import *
import numpy as np

N_VAR = 3
Q_RHO = 0
Q_U = 1
Q_P = 2

NG = 4
shape = (0, 0)
N = None

lo = NG
hi = None

def scratch_array(nc=1):
    return np.squeeze(np.zeros((N+NG*2, nc)))

class PPMInterpolant:

    def __init__(self, U, a, *, limit=True, chi_flat=None):
        self.shape = U.shape

        self.a = a
        self.limit = limit
        self.chi_flat = chi_flat

        self.aint = scratch_array()

        self.ap = scratch_array()
        self.am = scratch_array()
        self.a6 = scratch_array()

        self.initialized = False

    def construct_parabola(self):
        ib = lo-2
        ie = hi+1

        da0 = scratch_array()
        dap = scratch_array()

        da0[ib:ie+1] = 0.5 * (self.a[ib+1:ie+2] - self.a[ib-1:ie])
        dap[ib:ie+1] = 0.5 * (self.a[ib+2:ie+3] - self.a[ib:ie+1])

        if self.limit:
            dl = scratch_array()
            dr = scratch_array()
            dr[ib:ie+1] = self.a[ib+1:ie+2] - self.a[ib:ie+1]
            dl[ib:ie+1] = self.a[ib:ie+1] - self.a[ib-1:ie]

            da0 = np.where(dl * dr < 0, 0.0,
                           np.sign(da0) * np.minimum(np.abs(da0),
                                                     2.0 * np.minimum(np.abs(dl),
                                                                      np.abs(dr))))
            dl[:] = dr[:]
            dr[ib:ie+1] = self.a[ib+2:ie+3] - self.a[ib+1:ie+2]

            dap = np.where(dl * dr < 0, 0.0,
                           np.sign(dap) * np.minimum(np.abs(dap),
                                                     2.0 * np.minimum(np.abs(dl),
                                                                      np.abs(dr))))
        self.aint[ib:ie+1] = 0.5 * (self.a[ib:ie+1] + self.a[ib+1:ie+2]) - \
                             (1.0 / 6.0) * (dap[ib:ie+1] - da0[ib:ie+1])

        self.ap[:] = self.aint[:]
        self.am[1:] = self.ap[:-1]

        if self.limit:
            test = (self.ap - self.a) * (self.a - self.am) < 0

            da = self.ap - self.am
            testm = da * (self.a - 0.5 * (self.am + self.ap)) > da**2 / 6
            self.am[:] = np.where(test, self.a, np.where(testm, 3.0*self.a - 2.0*self.ap, self.am))

            testp = -da**2 / 6 > da * (self.a - 0.5 * (self.am + self.ap))
            self.ap[:] = np.where(test, self.a, np.where(testp, 3.0*self.a - 2.0*self.am, self.ap))

        if self.chi_flat is not None:
            self.am[:] = (1.0 - self.chi_flat[:]) * self.a[:] + self.chi_flat[:] * self.am[:]
            self.ap[:] = (1.0 - self.chi_flat[:]) * self.a[:] + self.chi_flat[:] * self.ap[:]

        self.a6 = 6.0 * self.a - 3.0 * (self.am + self.ap)
        self.initialized = True

def flattening_coefficient(p, u):
    smallp = 1.e-10
    z0 = 0.75
    z1 = 0.85
    delta = 0.33

    dp = scratch_array()
    dp[lo-2:hi+3] = p[lo-1:hi+4] - p[lo-3:hi+2]

    dp2 = scratch_array()
    dp2[lo-2:hi+3] = p[lo:hi+5] - p[lo-4:hi+1]

    z = np.abs(dp) / np.clip(np.abs(dp2), smallp, None)
    chi = np.clip(1.0 - (z - z0) / (z1 - z0), 0.0, 1.0)

    du = scratch_array()
    du[lo-2:hi+3] = u[lo-1:hi+4] - u[lo-3:hi+2]

    test = scratch_array()
    test[lo-2:hi+3] = np.abs(dp[lo-2:hi+3]) / \
        np.minimum(p[lo-3:hi+2],
                   p[lo-1:hi+4]) > delta

    chi = np.where(np.logical_and(test, du < 0), chi, 1.0)

    chi[lo-1:hi+2] = np.where(dp[lo-1:hi+2] > 0,
                                        np.minimum(chi[lo-1:hi+2],
                                                   chi[lo-2:hi+1]),
                                        np.minimum(chi[lo-1:hi+2],
                                                   chi[lo:hi+3]))
    return chi

def update_N():
    global N, hi
    N = shape[0]
    hi = NG + N - 1

def PPM(U, solver, dt, dx, N, X, bc_type='outflow', **kwargs):
    global shape
    shape = U.shape
    update_N()

    q = generate_gc(con2prim_grid(U), NG, bc_type)
    dx_extended = generate_gc(dx, NG, bc_type)

    chi = flattening_coefficient(q[:, Q_P], q[:, Q_U])

    q_parabola = []
    for ivar in range(N_VAR):
        q_parabola.append(PPMInterpolant(U, q[:, ivar],
                                                    limit=False, chi_flat=None))
        q_parabola[-1].construct_parabola()

    def interface_states():
        q_left = scratch_array(nc=N_VAR)
        q_right = scratch_array(nc=N_VAR)
        for i in range(lo-1, hi+1):
            for ivar in range(N_VAR):
                # q_right for cell i is the 'ap' value
                q_right[i, ivar] = q_parabola[ivar].ap[i]

                # q_left for cell i+1 is the 'am' value
                q_left[i+1, ivar] = q_parabola[ivar].am[i+1]

        return q_left, q_right

    q_left, q_right = interface_states()
    flux = np.zeros((N + 1, N_VAR))

    UL = prim2con_grid(q_left)
    UR = prim2con_grid(q_right)

    for i in range(N + 1):
        flux[i] = solver(UR[i+NG-1], UL[NG+i])
    
    dU = (1 / dx[:, np.newaxis]) * (flux[1:] - flux[:-1])

    return -(dU)
    