import os
import sys
current_dir = os.getcwd()
src_path = os.path.join(current_dir, 'reconstruction')

if src_path not in sys.path:
    sys.path.append(src_path)
import numpy as np
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
    
def dt_method(dt_type, dx_type, U, dt, **kwargs):
    # Credit: Migone 2007
    func = dx_method(dx_type)

    if dt_type == 'RK1':
        L = func(U, dt=dt, **kwargs)
        final_U = U + dt * L
       
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