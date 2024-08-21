import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

def kde_loo_cv(X, sigma_range):
    n = len(X)
    E_sigma = []
    
    for sigma in sigma_range:
        log_likelihood = 0
        for i in range(n):
            X_loo = np.delete(X, i)
            kde = gaussian_kde(X_loo, bw_method=sigma)
            log_likelihood += np.log(kde(X[i]))
        E_sigma.append(-log_likelihood / n)
    
    return E_sigma

# Given dataset
X = np.array([3.918, -6.35, -2.677, -3.003])

# Range of sigma values to try
sigma_range = np.logspace(-1, 1, 100)

# Compute E(sigma) for each sigma
E_sigma = kde_loo_cv(X, sigma_range)

# Plot the results
plt.figure(figsize=(10, 6))
plt.semilogx(sigma_range, E_sigma)
plt.xlabel('σ (Kernel Width)')
plt.ylabel('E(σ) - Average Negative Log-Likelihood')
plt.title('LOO Cross-Validation for KDE Bandwidth Selection')
plt.grid(True)
plt.show()

# Find the optimal sigma
optimal_sigma = sigma_range[np.argmin(E_sigma)]
print(f"Optimal σ: {optimal_sigma:.4f}")