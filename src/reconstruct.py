import os
import sys
current_dir = os.getcwd()
src_path = os.path.join(current_dir, 'reconstruction')

if src_path not in sys.path:
    sys.path.append(src_path)
import numpy as np
from misc import con2flux
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
    new_dx = []
    for x, y in zip(dx, refined_index):
        if y == 1:
            new_value = x / 2
            new_dx.append(new_value)
            new_dx.append(new_value)
        else:
            new_dx.append(x)

    return np.array(new_dx)

def new_dt_method(dt_type, dx_type, U, dt, refined_U, dx,  refined_index, **kwargs):
    # Credit: Migone 2007
    func = dx_method(dx_type)

    def calc_dU(flux, U):
        new_flux = []
        for i, bool in enumerate(refined_index):
            if bool == 1:
                new_flux.append(flux[i])
                new_flux.append(con2flux(U[i]))
            else:
                new_flux.append(flux[i])
        new_flux.append(flux[-1])

        new_flux = np.array(new_flux)
        #flux = np.insert(flux, np.where(refined_index == 1)[0] + 1, 0, axis=0)  
        new_dx = calc_new_dx(dx, refined_index)
        dU = (1 / new_dx[:, np.newaxis]) * (new_flux[1:] - new_flux[:-1])
        return -dU#-calc_new_dx(dU, refined_index)

    if dt_type == 'RK1':
        L = calc_dU(func(U, dt=dt, dx=dx, **kwargs), U=U)
        #print(refined_U.shape, L.shape, U.shape)
        final_U = refined_U + dt * L
       
    elif dt_type == 'RK2': # Two-stage SSPRK method
        L = func(U, dt=dt, **kwargs)
        U1 = U + dt * L
        L1 = func(U1, dt=dt, **kwargs)
        final_U = 0.5 * (U + U1 + dt * L1)
       
    elif dt_type == 'RK3': # Three-stage SSPRK method
        L = func(U, dt=dt, **kwargs)
        U1 = U + dt * L
        L1 = func(U1, dt=dt, **kwargs)
        U2 = 0.25 * (3 * U + U1 + dt * L1)
        L2 = func(U2, dt=dt, **kwargs)
        final_U = (1/3) * (U + 2 * U2 + 2 * dt * L2)
    else:
        raise ValueError("The entered temporal integration method has not been implemented or its coefficients are not defined.")
    return final_U

def dt_method(dt_type, dx_type, U, dt, dx, **kwargs):
    # Credit: Migone 2007
    func = dx_method(dx_type)

    def calc_dU(flux):
        dU = (1 / dx[:, np.newaxis]) * (flux[1:] - flux[:-1])

        return -dU

    if dt_type == 'RK1':
        L = calc_dU(func(U, dt=dt, dx=dx, **kwargs))
        final_U = U + dt * L
       
    elif dt_type == 'RK2': # Two-stage SSPRK method
        L = calc_dU(func(U, dt=dt, dx=dx, **kwargs))
        U1 = U + dt * L
        L1 = calc_dU(func(U1, dt=dt, dx=dx, **kwargs))
        final_U = 0.5 * (U + U1 + dt * L1)
       
    elif dt_type == 'RK3': # Three-stage SSPRK method
        L = calc_dU(func(U, dt=dt, dx=dx, **kwargs))
        U1 = U + dt * L
        L1 = calc_dU(func(U1, dt=dt, dx=dx, **kwargs))
        U2 = 0.25 * (3 * U + U1 + dt * L1)
        L2 = calc_dU(func(U2, dt=dt, dx=dx, **kwargs))
        final_U = (1/3) * (U + 2 * U2 + 2 * dt * L2)
    else:
        raise ValueError("The entered temporal integration method has not been implemented or its coefficients are not defined.")
    return final_U