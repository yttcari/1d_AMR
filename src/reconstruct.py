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
    func = dx_method(dx_type)

    if dt_type == 'RK1':
        gamma = [[0, 1]]
        beta = [1]
    elif dt_type == 'VL2': # Two-stage predictor-corrector method
        gamma = [[0, 1],
                 [0, 1]]
        beta = [0.5, 1]
    elif dt_type == 'RK2': # Two-stage SSPRK method
        gamma = [[0, 1],
                 [0.5, 0.5]]
        beta = [1, 0.5]
    elif dt_type == 'RK3': # Three-stage SSPRK method
        gamma = [[0, 1],
                 [0.25, 0.75],
                 [2/3, 1/3]]
        beta = [1, 0.25, 2/3]
    else:
        raise ValueError("The entered temporal integration method has not been implemented or its coefficients are not defined.")
    
    N_stage = len(gamma)
    
    u0 = U.copy()  
    u1 = U.copy()  
    
    u0 = U.copy()
    u1 = 0 * U  # Initialize u^(1) to zero
    
    for s in range(N_stage):
        if s == 0:
            delta_s = 1
        else:
            delta_s = 0
            
        u1_new = u1 + delta_s * u0
        
        spatial_term = func(u0, dt=dt, **kwargs)
        u0_new = gamma[s][0] * u0 + gamma[s][1] * u1_new + beta[s] * spatial_term
        
        u0 = u0_new
        u1 = u1_new
    
    return u0