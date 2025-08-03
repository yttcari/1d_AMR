import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def f_P_star(P_star, P_L, rho_L, u_L, c_L, P_R, rho_R, u_R, c_R, gamma, **kwargs):
    term1 = (P_star / P_L)**((gamma - 1) / (2 * gamma))
    u_rarefaction_side = u_L + (2 * c_L / (gamma - 1)) * (1 - term1)

    u_shock_side = u_R + (P_star - P_R) * np.sqrt((2 / ((gamma + 1) * rho_R)) / (P_star + (gamma - 1) / (gamma + 1) * P_R))

    return u_rarefaction_side - u_shock_side

def get_sod_solution(x, t_plot, rho_L, u_L, P_L, rho_R, u_R, P_R, gamma, x_diaphragm, **kwargs):
    c_L = np.sqrt(gamma * P_L / rho_L)
    c_R = np.sqrt(gamma * P_R / rho_R)

    P_star_guess = 0.5 * (P_L + P_R)
    P_star = fsolve(f_P_star, P_star_guess, args=(P_L, rho_L, u_L, c_L, P_R, rho_R, u_R, c_R, gamma))[0]

    u_star = u_L + (2 * c_L / (gamma - 1)) * (1 - (P_star / P_L)**((gamma - 1) / (2 * gamma)))

    rho_2 = rho_L * (P_star / P_L)**(1 / gamma)
    c_2 = np.sqrt(gamma * P_star / rho_2)

    rho_4 = rho_R * ((P_star / P_R) * (gamma + 1) + (gamma - 1)) / ((P_star / P_R) * (gamma - 1) + (gamma + 1))

    S_RH = u_L - c_L
    S_RF = u_star - c_2
    S_CD = u_star
    S_S = u_R + c_R * np.sqrt(((gamma + 1) / (2 * gamma)) * (P_star / P_R) + (gamma - 1) / (2 * gamma))

    X_RH = x_diaphragm + S_RH * t_plot
    X_RF = x_diaphragm + S_RF * t_plot
    X_CD = x_diaphragm + S_CD * t_plot
    X_S = x_diaphragm + S_S * t_plot

    rho = np.zeros_like(x)
    P = np.zeros_like(x)
    u = np.zeros_like(x)

    for i, xi in enumerate(x):
        if xi < X_RH:
            rho[i] = rho_L
            P[i] = P_L
            u[i] = u_L
        elif X_RH <= xi < X_RF:
            u_i = (2 / (gamma + 1)) * (c_L + ((xi - x_diaphragm) / t_plot) + (u_L * (gamma - 1) / 2))
            c_i = c_L - ((gamma - 1) / 2) * (u_i - u_L)
            P_i = P_L * (c_i / c_L)**(2 * gamma / (gamma - 1))
            rho_i = rho_L * (c_i / c_L)**(2 / (gamma - 1))
            rho[i] = rho_i
            P[i] = P_i
            u[i] = u_i
        elif X_RF <= xi < X_CD:
            rho[i] = rho_2
            P[i] = P_star
            u[i] = u_star
        elif X_CD <= xi < X_S:
            rho[i] = rho_4
            P[i] = P_star
            u[i] = u_star
        else:
            rho[i] = rho_R
            P[i] = P_R
            u[i] = u_R

    return np.array(rho), np.array(u), np.array(P)

def get_plane_wave_solution(x, t, A, L, rho0, u0, p0, gamma=5/3, **kwargs):
    k = 2 * np.pi / L
    cs = np.sqrt(gamma * p0 / rho0)
    phase = k * (x - (u0 + cs) * t)
    delta = A * np.sin(phase)
    return np.array([
        rho0 + delta,  # density
        u0 + delta,    # velocity
        p0 + delta     # pressure
    ])