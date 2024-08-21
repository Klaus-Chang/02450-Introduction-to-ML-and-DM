import numpy as np

# Given residuals from the table
residuals = np.array([0.00162, -0.00756, -0.0161, 0.0248, -0.00274])

# Calculate the mean squared error (MSE)
MSE = np.mean(residuals ** 2)

# Given data
lambda_value = 0.02
E_w = 0.00025757

# Known weights
w1 = 0.0318
w2 = 0.04
w3 = 0.000197

# Regularization term excluding w5
regularization_excluding_w5 = w1**2 + w2**2 + w3**2

# Solve for w5 using the given loss function value
# The total loss E(w) is composed of MSE and the regularization term.
# Rearranging the equation to solve for w5:
w5_squared = (E_w - MSE - lambda_value * regularization_excluding_w5) / lambda_value

# Since we expect w5 to be negative (based on the context of the problem),
# we take the negative root
w5 = -np.sqrt(w5_squared)

# Output the calculated MSE and w5
print(f"\nMSE, w5: {MSE, w5}")