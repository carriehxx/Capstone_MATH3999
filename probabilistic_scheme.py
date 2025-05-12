import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

step_length = np.array([])
error = np.array([])

T = 1.0
N_set = [20,30,40, 50,80,100] # 
num_paths = 100000
a = 0.04
a_bar = 0.09
mu = -0.2
mu_bar = 0.2
sigma_ref = np.sqrt(a)

num_bins_x = 20
num_bins_a = 28


for N in N_set:
    h = T / N
    print(f"current step length = {h:.4f}")

    X = np.zeros((N + 1, num_paths))
    A = np.zeros((N + 1, num_paths))
    for k in range(N):
        dW = np.sqrt(h) * np.random.randn(num_paths)
        X[k+1] = X[k] + sigma_ref * dW
        A[k+1] = A[k] + (X[k] + X[k+1]) * h / 2

    u = np.zeros_like(X)
    u[-1] = np.cos(X[-1] + A[-1])
    # print(f"the terminal time mean is {u[-1].mean()}")

    for k in range(N-1, -1, -1):
        X_k, A_k = X[k], A[k]
        X_k_next, A_k_next = X[k+1], A[k+1]
        u_next = u[k+1]

        x_bins = np.linspace(X_k.min(), X_k.max(), num_bins_x + 1)
        a_bins = np.linspace(A_k.min(), A_k.max(), num_bins_a + 1)
        x_idx = np.digitize(X_k, x_bins) - 1
        a_idx = np.digitize(A_k, a_bins) - 1

        E_u_next = np.zeros_like(X_k)
        E_H1 = np.zeros_like(X_k)
        E_H2 = np.zeros_like(X_k)

        if k == 0:
          dW = (X_k_next - X_k)/sigma_ref
          H_1 = (1/sigma_ref) * dW / h
          H_2 = (1/sigma_ref**2) * (dW**2 - h) / h**2
          E_u_next = u_next.mean()
          E_H1 = (u_next * H_1).mean()
          E_H2 = (u_next * H_2).mean()

        else:
          for i in range(num_bins_x):
              for j in range(num_bins_a):
                  mask = (x_idx == i) & (a_idx == j)
                  if np.sum(mask) < 10:
                      continue

                  X_masked = X_k[mask]
                  A_masked = A_k[mask]
                  u_next_masked = u_next[mask]

                  if k == 1 :
                      design = np.column_stack([np.ones_like(X_masked), X_masked, X_masked**2])
                  else:
                      design = np.column_stack([
                          np.ones_like(X_masked),
                          X_masked,
                          A_masked,
                          X_masked * A_masked,
                          X_masked**2,
                          A_masked**2
                      ])

                  dW_masked = (X_k_next[mask] - X_masked)/sigma_ref
                  H_1 = (1/sigma_ref) * dW_masked / h
                  H_2 = (1/sigma_ref**2) * (dW_masked**2 - h) / h**2

                  # model = LinearRegression(fit_intercept=False).fit(design, u_next_masked)
                  model = Ridge(alpha=1e-3, fit_intercept=False).fit(design, u_next_masked)
                  E_u_next[mask] = model.predict(design)

                  # model_H1 = LinearRegression(fit_intercept=False).fit(design, u_next_masked * H_1)
                  model_H1 = Ridge(alpha=1e-3, fit_intercept=False).fit(design, u_next_masked * H_1)
                  E_H1[mask] = model_H1.predict(design)

                  # model_H2 = LinearRegression(fit_intercept=False).fit(design, u_next_masked * H_2)
                  model_H2 = Ridge(alpha=1e-3, fit_intercept=False).fit(design, u_next_masked * H_2)
                  E_H2[mask] = model_H2.predict(design)

        z = E_H1
        gamma = E_H2
        f_val = (
            - (X_k - mu) * np.maximum(-np.sin(X_k - A_k), 0)
            + (X_k + mu_bar) * np.maximum(np.sin(X_k - A_k), 0)
            + (a/2) * np.maximum(np.cos(X_k - A_k), 0)
            - (a_bar/2) * np.maximum(-np.cos(X_k - A_k), 0)
        )

        max_a_term = np.where(gamma >= 0, a_bar * gamma, a * gamma) / 2
        min_mu = np.where(z >= 0, mu * z, mu_bar * z)
        F_total = min_mu + max_a_term + f_val - 0.5 * sigma_ref**2 * gamma

        u[k] = E_u_next + h * F_total
        # print(f"the mean for step {k} is {u[k].mean()}")

    step_length = np.append(step_length, h)
    error = np.append(error, abs(u[0].mean() - 1))
    print(f"N={N}, Error={error[-1]:.6f}")
