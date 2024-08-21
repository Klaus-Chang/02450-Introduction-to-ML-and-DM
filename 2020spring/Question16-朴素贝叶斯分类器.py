import numpy as np

# 二值化的数据，其中最后一列是 y_b
data = np.array([
    [0, 1, 1, 1, 0, 0],  # o1
    [1, 1, 1, 0, 0, 0],  # o2
    [1, 1, 1, 0, 0, 0],  # o3
    [1, 1, 1, 1, 0, 0],  # o4
    [1, 1, 1, 0, 0, 0],  # o5
    [0, 1, 1, 0, 1, 0],  # o6
    [0, 1, 1, 0, 0, 0],  # o7
    [1, 0, 0, 0, 1, 1],  # o8
    [0, 0, 1, 1, 0, 1],  # o9
    [0, 0, 0, 1, 1, 1],  # o10
    [1, 0, 1, 0, 1, 1]   # o11
])

# y_b = 1 的索引
yb_indices = [8, 9, 10]  # o9, o10, o11 的索引

# 提取 y_b = 1 的样本
data_yb_1 = data[yb_indices]

# 正则化因子 alpha
alpha = 1

# 计算 y_b = 1 的总次数
total_yb_1 = data_yb_1.shape[0]

# 计算 f2 = 1, f3 = 1, 且 yb = 1 的次数
#在计算count_f2_1_f3_1_yb_1时使用data_yb_1数组，而不是整个data数组。
count_f2_1_f3_1_yb_1 = np.sum((data_yb_1[:, 1] == 1) & (data_yb_1[:, 2] == 1))

# 计算概率 p(f2=1, f3=1 | yb=1)
prob = (count_f2_1_f3_1_yb_1 + alpha) / (total_yb_1 + 2 * alpha)
print(f"p(f2=1, f3=1 | yb=1) = {prob:.2f}")
