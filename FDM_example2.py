import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize_scalar

# Parameters
T = 1.0
step_length_set = np.array([0.1, 0.05, 1/30, 0.025, 0.02, 0.01, 0.005])  
delta_x_set = np.sqrt(step_length_set)/4 
delta_y_set = np.sqrt(step_length_set)/4    

k1 = -0.2
k2 = 0.2
a_lower = 0.04
a_upper= 0.09
b = 0.05

# Discretization ranges
x_min, x_max = -0.8, 0.8
y_min, y_max = -0.8, 0.8

def compute_G_a(a, d2_u_dx2, d_u_dx, b):
    term_inside = (np.sqrt(a) * d_u_dx + b / np.sqrt(a))
    max_term = np.maximum(-term_inside, 0)
    f_val_a = 0.5 * (max_term ** 2) - b * d_u_dx - (b**2) / (2 * a)
    G_a = 0.5 * a * d2_u_dx2 - f_val_a
    return G_a

def objective(a, d2_u_dx2, d_u_dx, b):
    return -compute_G_a(a, d2_u_dx2, d_u_dx, b)

n = len(step_length_set)
for i in range(n):
  step_length = step_length_set[i]
  delta_x = delta_x_set[i]
  delta_y = delta_y_set[i]
  print(f"current time step length is {step_length}, delta x is {delta_x}, and delta y is {delta_y}")

  # Grids
  time_grid = np.arange(0, T + step_length, step_length)
  x_grid = np.arange(x_min, x_max + delta_x, delta_x)
  y_grid = np.arange(y_min, y_max + delta_y, delta_y)

  # Initialize solution grid: u[t, x_idx, y_idx]
  u = np.zeros((len(time_grid), len(x_grid), len(y_grid)))

  # Terminal condition at T
  for i_x, x in enumerate(x_grid):
      for i_y, y in enumerate(y_grid):
          u[-1, i_x, i_y] = k1 + np.maximum(y - k1, 0) - np.maximum(y - k2, 0)

  # Dynamic programming backward iteration
  for i_t in reversed(range(len(time_grid) - 1)):
      t = time_grid[i_t]
      count = 0
      # print(f"Processing time step {t:.2f}")

      for i_x, x in enumerate(x_grid):
          for i_y, y in enumerate(y_grid):

              # Approximate Y_next (simplified integration)
              y_next = y + x * step_length
              y_plus_next = y + (x + delta_x + x) * step_length/2
              y_plus2_next = y + (x + 2*delta_x + x) * step_length/2
              y_minus_next = y + (x - delta_x + x) * step_length/2
              y_minus2_next = y + (x - 2*delta_x + x) * step_length/2

              # Find nearest y index for y_next (with boundary checks)
              i_y_next = np.clip(int(round((y_next - y_min) / delta_y)), 0, len(y_grid)-1)
              i_y_plus_next = np.clip(int(round((y_plus_next - y_min) / delta_y)), 0, len(y_grid)-1)
              i_y_plus2_next = np.clip(int(round((y_plus2_next - y_min) / delta_y)), 0, len(y_grid)-1)
              i_y_minus_next = np.clip(int(round((y_minus_next - y_min) / delta_y)), 0, len(y_grid)-1)
              i_y_minus2_next = np.clip(int(round((y_minus2_next - y_min) / delta_y)), 0, len(y_grid)-1)

              # discrete drivatives
              if i_x == 0:
                  d_u_dx = (u[i_t+1, i_x+1, i_y_plus_next] - u[i_t+1, i_x, i_y_next]  ) / (delta_x)
                  d2_u_dx2 = (u[i_t+1,i_x,i_y_next] -2*u[i_t+1,i_x+1,i_y_plus_next]
                                +u[i_t+1,i_x+2,i_y_plus2_next]) / delta_x**2

              elif i_x == len(x_grid)-1:
                  d_u_dx = (u[i_t+1, i_x, i_y_next] - u[i_t+1, i_x-1, i_y_minus_next] ) / (delta_x)
                  d2_u_dx2 = (u[i_t+1,i_x,i_y_next] - 2*u[i_t+1,i_x-1,i_y_minus_next]
                                + u[i_t+1,i_x-2,i_y_minus2_next]) / delta_x**2
              else:
                  d_u_dx = (u[i_t+1, i_x+1, i_y_plus_next] - u[i_t+1, i_x, i_y_next]) / (delta_x)
                  d2_u_dx2 = (u[i_t+1, i_x+1, i_y_plus_next] - 2*u[i_t+1, i_x, i_y_next] + u[i_t+1, i_x-1, i_y_minus_next]) / (delta_x**2)

              result = minimize_scalar(
                      objective,
                      bounds=(a_lower, a_upper),
                      args=(d2_u_dx2, d_u_dx, b),
                      method='bounded'
                  )
              if result.success:
                  optimal_a = result.x
                  maximized_G = -result.fun
              else:
                  print("optimization failed")

              u[i_t, i_x, i_y] = u[i_t+1, i_x, i_y_next] + step_length * maximized_G

  interp_func = RegularGridInterpolator((x_grid, y_grid), u[0], method='linear')
  u_interp = interp_func([[0, 0]])[0]
  print(f"the solution is = {u_interp} for step_length {step_length} and delta_y {delta_y}")