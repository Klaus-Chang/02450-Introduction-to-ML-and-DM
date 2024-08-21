import numpy as np
from itertools import combinations

def rand_index(confusion_matrix):
    n = np.sum(confusion_matrix)
    sum_squares = np.sum(confusion_matrix ** 2)
    row_sums = np.sum(confusion_matrix, axis=1)
    col_sums = np.sum(confusion_matrix, axis=0)
    
    a = (sum_squares - n) / 2.0
    b = (np.sum(row_sums ** 2) - sum_squares) / 2
    c = (np.sum(col_sums ** 2) - sum_squares) / 2
    d = (n ** 2 + sum_squares - np.sum(row_sums ** 2) - np.sum(col_sums ** 2)) / 2
    
    return (a + d) / (a + b + c + d)

# 从图中提取的混淆矩阵
confusion_matrix = np.array([
    [114, 0, 32],
    [0, 119, 0],
    [8, 0, 80]
])

ri = rand_index(confusion_matrix)
print(f"Rand Index: {ri:.2f}")