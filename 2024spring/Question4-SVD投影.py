import numpy as np

# 定义表中给出的 1% 和 99% 百分位数对应的值
percentiles = {
    "x1": {"low": -1.19, "high": 1.21},
    "x2": {"low": -1.6, "high": 2.59},
    "x3": {"low": -1.76, "high": 1.48},
    "x4": {"low": -0.7, "high": 3.21},
    "x5": {"low": -1.21, "high": 2.25}
}

# 根据题意定义特征组合，使用表中的 low 和 high 值
combination_A = np.array([percentiles["x1"]["low"], 0, percentiles["x3"]["low"], percentiles["x4"]["low"], percentiles["x5"]["high"]])
combination_B = np.array([percentiles["x1"]["high"], 0, percentiles["x3"]["high"], percentiles["x4"]["high"], percentiles["x5"]["high"]])
combination_C = np.array([percentiles["x1"]["high"], percentiles["x2"]["high"], percentiles["x3"]["low"], 0, 0])
combination_D = np.array([percentiles["x1"]["low"], percentiles["x2"]["high"], percentiles["x3"]["high"], 0, 0])

# V矩阵的前三列
V = np.array([
[-0.47, 0.52, -0.24, 0.54, 0.4],
[0., 0.44, 0.9, -0.04, 0.02],
[-0.31, -0.73, 0.37, 0.41, 0.27],
[-0.58, -0.04, -0.01, -0.73, 0.36],
[-0.6, 0.01, 0.01, 0.08, -0.8]
])

# 计算投影值
projection_A = np.dot(combination_A, V)
projection_B = np.dot(combination_B, V)
projection_C = np.dot(combination_C, V)
projection_D = np.dot(combination_D, V)

# 输出投影结果
print("Projection for A:", projection_A)
print("Projection for B:", projection_B)
print("Projection for C:", projection_C)
print("Projection for D:", projection_D)
