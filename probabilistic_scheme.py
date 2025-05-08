import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

# Parameters
T = 1.0                  # Terminal time
N = 30                   # Number of time steps
h = T / N                # Time step size
num_paths = 100000       # Number of simulated paths
a = 0.04                  # Lower bound for a
a_bar = 0.09             # Upper bound for a
mu = -0.2                  # Lower bound for μ
mu_bar = 0.2             # Upper bound for μ

# Simulate reference paths for X and A using a neutral diffusion
# Note: The actual control on 'a' is applied in the F term, not in path simulation
X = np.zeros((N + 1, num_paths))
A = np.zeros((N + 1, num_paths))
sigma_ref = np.sqrt(a) # sigma_ref = 0.2

for k in range(N):
    dW =  np.sqrt(h) * np.random.randn(num_paths)
    X[k + 1] = X[k] + sigma_ref * dW
    A[k + 1] = A[k] + (X[k] + X[k+1]) * h / 2


# Terminal condition u(T, X_T, A_T) = cos(X_T + A_T)
u = np.zeros_like(X)
u[-1] = np.cos(X[-1] + A[-1])
print(f"time step {N}")
print(u[-1].mean())

# Hypercube parameters
num_bins_x, num_bins_a = 20, 28

# Backward iteration
for k in range(N - 1, -1, -1):
    print(f"Time step {k}")
    X_k, A_k = X[k], A[k]
    X_k_next, A_k_next = X[k+1], A[k+1]
    u_next = u[k + 1]

    x_min_k, x_max_k = X_k_next.min(), X_k_next.max()
    a_min_k, a_max_k = A_k_next.min(), A_k_next.max()
    x_bins = np.linspace(x_min_k, x_max_k, num_bins_x + 1)
    a_bins = np.linspace(a_min_k, a_max_k, num_bins_a + 1)

    # Compute H_1 and H_2 (sigma_ref is fixed for path simulation)
    dW = (X[k + 1] - X[k]) / sigma_ref
    sigma_inv = 1 / sigma_ref
    H_1 = sigma_inv * dW / h
    H_2 = sigma_inv**2 * (dW**2 - h) / h**2

    # Products for regression
    product_H1 = u_next * H_1
    product_H2 = u_next * H_2

    # Bin indices for current (X_k, A_k)
    x_idx = np.digitize(X_k_next, x_bins) - 1
    a_idx = np.digitize(A_k_next, a_bins) - 1

    # Initialize expectations
    E_H1 = np.zeros(num_paths)
    E_H2 = np.zeros(num_paths)
    E_u_next = np.zeros(num_paths)

    # Loop over hypercubes
    for i in range(num_bins_x):
        for j in range(num_bins_a):
            mask = (x_idx == i) & (a_idx == j)
            if not np.any(mask):
                continue

            # Regression basis: [1, X, A, X², A², XA]
            X_masked = X_k_next[mask]
            A_masked = A_k_next[mask]
            # design = np.column_stack([np.ones_like(X_masked), X_masked, A_masked, X_square, A_square, XA])

            if k == N-1:
                design = np.column_stack([np.ones_like(X_masked), X_masked, X_masked**2])
            else:
                design = np.column_stack([np.ones_like(X_masked), X_masked, A_masked,
                                        X_masked*A_masked, X_masked**2, A_masked**2])
            if k == 0:
                E_u_next = np.mean(u_next, axis=0)
                E_H1 = np.mean(product_H1, axis=0)
                E_H2 = np.mean(product_H2, axis=0)
            else:
                # model = Ridge(alpha=1, fit_intercept=False).fit(design, product_H1[mask])
                # E_H1[mask] = model.predict(design)

                # model = Ridge(alpha=1, fit_intercept=False).fit(design, product_H2[mask])
                # E_H2[mask] = model.predict(design)

                # model = Ridge(alpha=1, fit_intercept=False).fit(design, u_next[mask])
                # E_u_next[mask] = model.predict(design)

                # Regression for E[product_H1 | X, A]
                model = LinearRegression(fit_intercept=False).fit(design, product_H1[mask])
                E_H1[mask] = model.predict(design)

                # Regression for E[product_H2 | X, A]
                model = LinearRegression(fit_intercept=False).fit(design, product_H2[mask])
                E_H2[mask] = model.predict(design)

                # Regression for E[u_next | X, A]
                model = LinearRegression(fit_intercept=False).fit(design, u_next[mask])
                E_u_next[mask] = model.predict(design)

    z = E_H1 # D1
    gamma = E_H2 # D2

    f_val = (
         - (X_k - mu) * np.maximum(-np.sin(X_k - A_k), 0)
        + (X_k + mu_bar) * np.maximum(np.sin(X_k - A_k), 0)
        + (a / 2) * np.maximum(np.cos(X_k - A_k), 0)
        - (a_bar / 2) * np.maximum(-np.cos(X_k - A_k), 0)
    )

    max_a_term = np.where(gamma >= 0, a_bar*gamma/2, a*gamma/2)
    min_mu = np.where(z >= 0, mu * z, mu_bar * z)
    F_total =  (min_mu +max_a_term + f_val) - 0.5 * sigma_ref**2 * gamma
    u[k] = E_u_next + h * F_total
    print(u[k].mean())

# Solution at t=0 is u[0].mean()
print("Approximate solution at t=0:", u[0].mean())
print(f"the numerical error is: {abs(u[0].mean()-1)}")