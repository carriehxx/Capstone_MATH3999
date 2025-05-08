import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import RegularGridInterpolator

# Parameters
T = 1.0
step_length = 0.02  # Time step Δt, try [0.05, 0.0333, 0.0250, 0.0200, 0.0125, 0.0100, 0.0050]
delta_x = np.sqrt(step_length)/2  # Spatial step for X
delta_y = 0.05      # Step for path integral Y

mu_lower = -0.2
mu_upper = 0.2
a_lower = 0.04
a_upper = 0.09

# Discretization ranges
x_min, x_max = -3.0, 3.0
y_min, y_max = -10.0, 10.0

# Grids
time_grid = np.arange(0, T + step_length, step_length)
x_grid = np.arange(x_min, x_max + delta_x, delta_x)
y_grid = np.arange(y_min, y_max + delta_y, delta_y)

# Initialize solution grid: u[t, x_idx, y_idx]
u = np.zeros((len(time_grid), len(x_grid), len(y_grid)))
delta_t = []
error = []

# Terminal condition at T
for i_x, x in enumerate(x_grid):
    for i_y, y in enumerate(y_grid):
        u[-1, i_x, i_y] = math.cos(x + y)

# Dynamic programming backward iteration
for i_t in reversed(range(len(time_grid) - 1)):
    t = time_grid[i_t]
    print(f"Processing time step {t:.2f}")
    
    for i_x, x in enumerate(x_grid):
        for i_y, y in enumerate(y_grid):
            
            # Approximate Y_next (simplified integration)
            y_next = y + x * step_length
            y_plus_next = y + (x + delta_x + x) * step_length/2
            y_plus2_next = y + (x + 2*delta_x + x) * step_length/2
            y_plus3_next = y + (x + 3*delta_x + x) * step_length/2
            y_minus_next = y + (x - delta_x + x) * step_length/2
            y_minus2_next = y + (x - 2*delta_x + x) * step_length/2
            y_minus3_next = y + (x - 3*delta_x + x) * step_length/2

            # Find nearest y index for y_next (with boundary checks)
            i_y_next = np.clip(int(round((y_next - y_min) / delta_y)), 0, len(y_grid)-1)
            i_y_plus_next = np.clip(int(round((y_plus_next - y_min) / delta_y)), 0, len(y_grid)-1)
            i_y_plus2_next = np.clip(int(round((y_plus2_next - y_min) / delta_y)), 0, len(y_grid)-1)
            i_y_plus3_next = np.clip(int(round((y_plus3_next - y_min) / delta_y)), 0, len(y_grid)-1)
            i_y_minus_next = np.clip(int(round((y_minus_next - y_min) / delta_y)), 0, len(y_grid)-1)
            i_y_minus2_next = np.clip(int(round((y_minus2_next - y_min) / delta_y)), 0, len(y_grid)-1)
            i_y_minus3_next = np.clip(int(round((y_minus3_next - y_min) / delta_y)), 0, len(y_grid)-1)


            # discrete drivatives
            if i_x == 0: 
                d_u_dx = (-3*u[i_t+1, i_x, i_y_next] + 4*u[i_t+1, i_x+1, i_y_plus_next] - u[i_t+1, i_x+2, i_y_plus2_next]) / (2*delta_x)
                # forward difference
                d2_u_dx2 = (2*u[i_t+1,i_x,i_y_next] -5*u[i_t+1,i_x+1,i_y_plus_next]
                               +4*u[i_t+1,i_x+2,i_y_plus2_next] - u[i_t+1,i_x+3,i_y_plus3_next]) / delta_x**2

            elif i_x == len(x_grid)-1:
                d_u_dx = (3*u[i_t+1, i_x, i_y_next] - 4*u[i_t+1, i_x-1, i_y_minus_next] + u[i_t+1, i_x-2, i_y_minus2_next]) / (2*delta_x)
                # backward difference
                d2_u_dx2 = (2*u[i_t+1,i_x,i_y_next] -5*u[i_t+1,i_x-1,i_y_minus_next]
                               +4*u[i_t+1,i_x-2,i_y_minus2_next] - u[i_t+1,i_x-3,i_y_minus3_next]) / delta_x**2
            else:
                # First derivative ∂u/∂x
                d_u_dx = (u[i_t+1, i_x+1, i_y_plus_next] - u[i_t+1, i_x, i_y_next]) / (delta_x)
                # Second derivative ∂²u/∂x²
                d2_u_dx2 = (u[i_t+1, i_x+1, i_y_plus_next] - 2*u[i_t+1, i_x, i_y_next] + u[i_t+1, i_x-1, i_y_minus_next]) / (delta_x**2)

            mu = mu_lower if d_u_dx >= 0 else mu_upper
            a = a_upper if d2_u_dx2 >= 0 else a_lower
            f_val = - (x - mu_lower)*max(-math.sin(x - y), 0) \
                    + (x + mu_upper)*max(math.sin(x - y), 0) \
                    + (a_lower/2)*max(math.cos(x - y), 0) \
                    - (a_upper/2)*max(-math.cos(x - y), 0)
            
            u[i_t, i_x, i_y] = u[i_t+1, i_x, i_y_next] + step_length * (mu * d_u_dx + 0.5 * a * d2_u_dx2 + f_val) 


interp_func = RegularGridInterpolator((x_grid, y_grid), u[0], method='linear')
u_interp = interp_func([[0, 0]])[0]
print(f"the solution error is = {abs(u_interp-1)}")
delta_t.append(step_length)
error.append(float(abs(u_interp-1)))
error_scaled = error * 1e3

plt.figure(figsize=(7, 4))
plt.plot(delta_t, error_scaled, marker='o', linestyle='-', color='b', label='Numerical Error')

plt.xlabel('Step Length')
plt.ylabel('Error ($\\times 10^{-3}$)')
plt.title('Numerical Error for Finite Difference Method (Numerical Example 1)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()