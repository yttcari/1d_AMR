import os
import sys
current_dir = os.getcwd()
src_path = os.path.join(current_dir, 'reconstruction')

if src_path not in sys.path:
    sys.path.append(src_path)
import numpy as np
from misc import con2flux, con2prim_grid, prim2con
from reconstruction import ppm, MUSCL, godunov

def dx_method(dx_type):
    if dx_type == 'godunov':
        return godunov.godunov
    if dx_type == 'MUSCL':
        return MUSCL.MUSCL
    if dx_type == 'PPM':
        return ppm.PPM
    else:
        raise ValueError("The reconstruction method is not implemented.")
    
def calc_new_dx(dx, refined_index):
    # dx: dx before refinement
    new_dx = []
    for x, y in zip(dx, refined_index):
        if y == 1:
            new_value = x / 2
            new_dx.append(new_value)
            new_dx.append(new_value)
        else:
            new_dx.append(x)

    return np.array(new_dx)

def calc_new_dU(flux, U, refined_index, dx):
        # dx: dx after refinement
        # U: U before refinement
        new_flux = []
        for i, bool in enumerate(refined_index):
            if bool == 1:
                new_flux.append(flux[i])
                new_flux.append(con2flux(U[i]))
            else:
                new_flux.append(flux[i])
        new_flux.append(flux[-1])

        new_flux = np.array(new_flux)
        dU = (1 / dx[:, np.newaxis]) * (new_flux[1:] - new_flux[:-1])
        return -dU

def calc_dU(flux, dx):
    dU = (1 / dx[:, np.newaxis]) * (flux[1:] - flux[:-1])

    return -dU

def coarse_from_index(q, coarse_index):
    assert np.sum(coarse_index) % 2 == 0

    if q.ndim == 1:
        coarsed_q = np.zeros((len(q) - int(np.sum(coarse_index)/2)))
    else:
        coarsed_q = np.zeros((len(q) - int(np.sum(coarse_index)/2), q.shape[1]))
    i = 0 
    j = 0
    #print(coarse_index)
    while i < len(coarse_index):
        #print(i)
        if coarse_index[i] == 1:
            coarsed_q[j] = (q[i] + q[i+1]) / 2
            i += 1 # assumed binary division
        else:
            coarsed_q[j] = q[i]
        i += 1   
        j += 1 
    return coarsed_q

def euler_and_coarse(func, dt, dx, old_N, old_X, U, new_dx, refined_index, refined_U, grid, **kwargs):
    L = calc_new_dU(func(U, dt=dt, dx=dx, N=old_N, X=old_X, **kwargs), U=U, dx=new_dx, refined_index=refined_index)
    #print(refined_U.shape, L.shape, U.shape)
    final_U = refined_U + dt * L

    grid.update(con2prim_grid(final_U))
    grid.new_flag(**kwargs)
    coarse_index = grid.coarse()

    return grid, final_U, L, coarse_index

def new_dt_method(dt_type, dx_type, U, dt, refined_U, dx, refined_index, old_N, old_X, grid, refined_N=None, refined_X=None, **kwargs):
    # Credit: Migone 2007
    func = dx_method(dx_type)
    new_dx = calc_new_dx(dx, refined_index)

    if dt_type == 'RK1':
        grid, _, _, _ = euler_and_coarse(func, dt, dx, old_N, old_X, U, new_dx, refined_index, refined_U, grid, **kwargs)
    
    elif dt_type == 'RK2': # Two-stage SSPRK method
        """
        Assumed RK1 is performed to refine the requre cell
        the parameter passed is the new grid structure
        """
        grid, U1, L, coarse_index = euler_and_coarse(func, dt, dx, old_N, old_X, U, new_dx, refined_index, refined_U, grid, **kwargs)

        active_cells = grid.get_all_active_cells()
        
        refined_U = coarse_from_index(refined_U, coarse_index)
        U1 = coarse_from_index(U1, coarse_index)
        refined_X = np.array([c.x for c in active_cells])
        new_dx = np.array([c.dx for c in active_cells])

        L1 = calc_dU(func(U1, dt=dt, dx=new_dx, N=len(U1), X=refined_X, **kwargs), dx=new_dx)
        final_U = 0.5 * (refined_U + U1 + dt * L1)

        grid.update(con2prim_grid(final_U))

        #grid.update(U1)
        #raise SystemError()       
    elif dt_type == 'RK3': # Three-stage SSPRK method
        L = func(U, dt=dt, **kwargs)
        U1 = U + dt * L
        L1 = func(U1, dt=dt, **kwargs)
        U2 = 0.25 * (3 * U + U1 + dt * L1)
        L2 = func(U2, dt=dt, **kwargs)
        final_U = (1/3) * (U + 2 * U2 + 2 * dt * L2)
    else:
        raise ValueError("The entered temporal integration method has not been implemented or its coefficients are not defined.")
    return grid

def dt_method(dt_type, dx_type, U, dt, dx, **kwargs):

    # Credit: Migone 2007
    func = dx_method(dx_type)

    if dt_type == 'RK1':
        L = calc_dU(func(U, dt=dt, dx=dx, **kwargs), dx=dx)
        final_U = U + dt * L
       
    elif dt_type == 'RK2': # Two-stage SSPRK method
        L = calc_dU(func(U, dt=dt, dx=dx, **kwargs), dx=dx)
        U1 = U + dt * L
        L1 = calc_dU(func(U1, dt=dt, dx=dx, **kwargs), dx=dx)
        final_U = 0.5 * (U + U1 + dt * L1)
       
    elif dt_type == 'RK3': # Three-stage SSPRK method
        L = calc_dU(func(U, dt=dt, dx=dx, **kwargs), dx=dx)
        U1 = U + dt * L
        L1 = calc_dU(func(U1, dt=dt, dx=dx, **kwargs), dx=dx)
        U2 = 0.25 * (3 * U + U1 + dt * L1)
        L2 = calc_dU(func(U2, dt=dt, dx=dx, **kwargs), dx=dx)
        final_U = (1/3) * (U + 2 * U2 + 2 * dt * L2)
    else:
        raise ValueError("The entered temporal integration method has not been implemented or its coefficients are not defined.")
    return final_U