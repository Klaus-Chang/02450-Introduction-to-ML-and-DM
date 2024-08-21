# Importing necessary libraries
import numpy as np
from scipy import stats

# Given data
A_losses = np.array([17, 21, 19, 20, 22])
B_losses = np.array([21, 21, 23, 21, 22])

# Calculate the differences between A and B
differences_A_B = A_losses - B_losses

# Perform a paired t-test
t_statistic, p_value = stats.ttest_rel(A_losses, B_losses)

print(f"\nt_statistic, p_value: {t_statistic, p_value}")