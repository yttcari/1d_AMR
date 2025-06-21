# 1: FVM in Heat Equation
Consider Heat equation
$\frac{\partial T}{\partial t} = \alpha \frac{\partial^2T}{\partial x^2}$

Integrating both sides: $\frac{\partial}{\partial t}\int^{x_{i+1/2}}_{x_{i-1/2}} Tdx= \int \frac{\partial^2T}{\partial x^2} dx
=\frac{\partial T_{i+1/2}}{\partial x}-\frac{\partial T_{i-1/2}}{\partial x}$

Also, at the same time, mean temperature in the cell $\bar{T}=\frac{1}{\Delta x}\int^{x_{i+1/2}}_{x_{i-1/2}} Tdx$

Therefore, $\frac{\partial \bar{T}}{\partial t}=\frac{1}{\Delta x}(\frac{\partial T_{i+1/2}}{\partial x}-\frac{\partial T_{i-1/2}}{\partial x})$

The spatial dervative at the face can be simply obtained by central difference method: $\frac{\partial T_{i-1/2}}{\partial x} = \frac{T_{i}-T_{i-1}}{\Delta x}$
