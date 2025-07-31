import os
import sys
current_dir = os.getcwd()
src_path = os.path.join(current_dir, 'reconstruction')

if src_path not in sys.path:
    sys.path.append(src_path)

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