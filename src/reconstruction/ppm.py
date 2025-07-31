from misc import *
import numpy as np

NG = 4
Q_RHO = 0
Q_U = 1
Q_P = 2

class PPMINterpolant:
    def __init__(self, a, lo, hi, chi):
        # a is single variable array that has shape N + NG*2

        self.a = a

        self.aint = np.zeros_like(self.a)
        self.ap = np.zeros_like(self.a)
        self.am = np.zeros_like(self.a)
        self.a6 = np.zeros_like(self.a)

        self.lo = lo
        self.hi = hi

        self.chi = chi

    def construct_parabola(self):
        ib = self.lo-2
        ie = self.hi+1

        da0 = np.zeros_like(self.a)
        dap = np.zeros_like(self.a)

        # 1/2 (a_{i+1} - a_{i-1})
        da0[ib:ie+1] = 0.5 * (self.a[ib+1:ie+2] - self.a[ib-1:ie])

        # 1/2 (a_{i+2} - a_{i})
        dap[ib:ie+1] = 0.5 * (self.a[ib+2:ie+3] - self.a[ib:ie+1])

        # van-Leer slopes
        dl = np.zeros_like(self.a)
        dr = np.zeros_like(self.a)
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

        # now the parabola coefficients
        self.ap[:] = self.aint[:]
        self.am[1:] = self.ap[:-1]

        test = (self.ap - self.a) * (self.a - self.am) < 0

        da = self.ap - self.am
        testm = da * (self.a - 0.5 * (self.am + self.ap)) > da**2 / 6
        self.am[:] = np.where(test, self.a, np.where(testm, 3.0*self.a - 2.0*self.ap, self.am))

        testp = -da**2 / 6 > da * (self.a - 0.5 * (self.am + self.ap))
        self.ap[:] = np.where(test, self.a, np.where(testp, 3.0*self.a - 2.0*self.am, self.ap))

        self.am[:] = (1.0 - self.chi[:]) * self.a[:] + self.chi[:] * self.am[:]
        self.ap[:] = (1.0 - self.chi[:]) * self.a[:] + self.chi[:] * self.ap[:]

        self.a6 = 6.0 * self.a - 3.0 * (self.am + self.ap)

    def integrate(self, sigma):
        Ip = np.zeros_like(self.a)
        Ip[:] = np.where(sigma <= 0.0, self.ap,
                         self.ap - 0.5 * np.abs(sigma) *
                           (self.ap - self.am - (1.0 - (2.0/3.0) * np.abs(sigma)) * self.a6))

        Im = np.zeros_like(self.a)
        Im[:] = np.where(sigma >= 0.0, self.am,
                         self.am + 0.5 * np.abs(sigma) *
                           (self.ap - self.am + (1.0 - (2.0/3.0) * np.abs(sigma)) * self.a6))

        return Im, Ip


def PPM(U, solver, dt, dx, N, X, **kwargs):
    nvar = U.shape[1]
    q = (np.pad(U, ((NG, NG), (0, 0)), 'edge'))
    dx_extended = np.pad(dx, ((NG, NG)), 'edge')

    lo = NG
    hi = NG + N -1

    def flattening_coefficient(p, u):

        smallp = 1.e-10
        z0 = 0.75
        z1 = 0.85
        delta = 0.33

        # dp = p_{i+1} - p_{i-1}
        dp = np.zeros_like(p)
        dp[lo-2:hi+3] = p[lo-1:hi+4] - p[lo-3:hi+2]

        # dp2 = p_{i+2} - p_{i-2}
        dp2 = np.zeros_like(p)
        dp2[lo-2:hi+3] = p[lo:hi+5] - p[lo-4:hi+1]

        z = np.abs(dp) / np.clip(np.abs(dp2), smallp, None)

        chi = np.clip(1.0 - (z - z0) / (z1 - z0), 0.0, 1.0)

        # du = u_{i+1} - u_{i-1}
        du = np.zeros_like(u)
        du[lo-2:hi+3] = u[lo-1:hi+4] - u[lo-3:hi+2]

        # construct |dp_i| / min(p_{i+1}, p_{i-1})
        test = np.zeros_like(u)
        test[lo-2:hi+3] = np.abs(dp[lo-2:hi+3]) / \
            np.minimum(p[lo-3:hi+2],
                    p[lo-1:hi+4]) > delta

        chi = np.where(np.logical_and(test, du < 0), chi, 1.0)

        # combine chi with the neighbor, following the sign of the pressure jump
        chi[lo-1:hi+2] = np.where(dp[lo-1:hi+2] > 0,
                                            np.minimum(chi[lo-1:hi+2],
                                                    chi[lo-2:hi+1]),
                                            np.minimum(chi[lo-1:hi+2],
                                                    chi[lo:hi+3]))
        return chi
    
    chi = flattening_coefficient(q[:, Q_P], q[:, Q_U])

    q_parabola = []
    for ivar in range(nvar):
        q_parabola.append(PPMINterpolant(q[:, ivar], lo=lo, hi=hi, chi=chi))
        q_parabola[-1].construct_parabola()

    # interface states
    cs = np.sqrt(GAMMA * q[:, Q_P] / q[:, Q_RHO])

    Ip = np.zeros((N+NG*2, 3, nvar))
    Im = np.zeros((N+NG*2, 3, nvar))

    for iwave, sgn in enumerate([-1, 0, 1]):
        sigma = (q[:, Q_U] + sgn * cs) * dt / dx_extended

        for ivar in range(nvar):
            Im[:, iwave, ivar], Ip[:, iwave, ivar] = q_parabola[ivar].integrate(sigma)

    q_left = np.zeros_like(q)
    q_right = np.zeros_like(q)

    for i in range(lo-1, hi+2):

        q_ref_m = Im[i, 0, :]

        # build eigensystem
        ev, lvec, rvec = eigen(q_ref_m[Q_RHO],
                                       q_ref_m[Q_U],
                                       q_ref_m[Q_P],
                                       GAMMA)

        beta_xm = np.zeros(3)
        for iwave in range(3):
            dq = q_ref_m - Im[i, iwave, :]
            beta_xm[iwave] = lvec[iwave, :] @ dq

        q_right[i, :] = q_ref_m[:]
        for iwave in range(3):
            if ev[iwave] <= 0:
                q_right[i, :] -= beta_xm[iwave] * rvec[iwave, :]

        q_ref_p = Ip[i, 2, :]


        ev, lvec, rvec = eigen(q_ref_p[Q_RHO],
                                       q_ref_p[Q_U],
                                       q_ref_p[Q_P],
                                       GAMMA)

        beta_xp = np.zeros(3)
        for iwave in range(3):
            dq = q_ref_p - Ip[i, iwave, :]
            beta_xp[iwave] = lvec[iwave, :] @ dq

        q_left[i+1, :] = q_ref_p[:]
        for iwave in range(3):
            if ev[iwave] >= 0:
                q_left[i+1, :] -= beta_xp[iwave] * rvec[iwave, :]

    flux = np.zeros((N+1, nvar))
    for i in range(lo, hi+2):
        flux[i-NG] = (solver(q_left[i], q_right[i]))

    dU = (dt / dx[:, np.newaxis]) * (flux[1:] - flux[:-1])

    return -dU