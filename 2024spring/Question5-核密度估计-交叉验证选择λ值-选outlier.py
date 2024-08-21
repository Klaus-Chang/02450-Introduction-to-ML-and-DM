import numpy as np
from scipy.stats import norm

# Given data points
X = np.array([-0.7, 1.1, 1.8])

# Potential outliers to evaluate
potential_outliers = np.array([-0.9, -0.3, 0.6, 1.2])

# Possible lambda values
lambdas = [0.5, 1]

def kde(x, X, lambda_):
    return np.mean(norm.pdf((x - X) / lambda_)) / lambda_

def loocv(X, lambda_):
    n = len(X)
    log_likelihood = 0
    for i in range(n):
        x_i = X[i]
        X_without_i = np.delete(X, i)
        kde_at_x_i = kde(x_i, X_without_i, lambda_)
        log_likelihood += np.log(kde_at_x_i)
    return log_likelihood

# Step 1: LOOCV to select best lambda
loocv_results = {lambda_: loocv(X, lambda_) for lambda_ in lambdas}
best_lambda = max(loocv_results, key=loocv_results.get)
print(f"LOOCV results: {loocv_results}")
print(f"Best lambda: {best_lambda}")

# Step 2: Calculate KDE for potential outliers using best lambda
kde_values = [kde(x, X, best_lambda) for x in potential_outliers]

print("\nKDE values for potential outliers:")
for x, kde_value in zip(potential_outliers, kde_values):
    print(f"KDE value for x* = {x}: {kde_value:.6f}")

# Step 3: Identify potential outlier
min_kde_index = np.argmin(kde_values)
potential_outlier = potential_outliers[min_kde_index]
print(f"\nPoint most likely to be a potential outlier: {potential_outlier}")