import numpy as np

# 定义正确的距离矩阵
distance_matrix = np.array([
    [0.0, 1.7, 1.4, 0.4, 2.2, 3.7, 5.2, 0.2, 4.3, 6.8, 6.0],
    [1.7, 0.0, 1.0, 2.0, 1.3, 2.6, 4.5, 1.8, 3.2, 5.9, 5.2],
    [1.4, 1.0, 0.0, 1.7, 0.9, 2.4, 4.1, 1.5, 3.0, 5.5, 4.8],
    [0.4, 2.0, 1.7, 0.0, 2.6, 4.0, 5.5, 0.3, 4.6, 7.1, 6.3],
    [2.2, 1.3, 0.9, 2.6, 0.0, 1.7, 3.4, 2.4, 2.1, 4.8, 4.1],
    [3.7, 2.6, 2.4, 4.0, 1.7, 0.0, 2.0, 3.8, 1.6, 3.3, 2.7],
    [5.2, 4.5, 4.1, 5.5, 3.4, 2.0, 0.0, 5.4, 2.5, 1.6, 0.9],
    [0.2, 1.8, 1.5, 0.3, 2.4, 3.8, 5.4, 0.0, 4.4, 6.9, 6.1],
    [4.3, 3.2, 3.0, 4.6, 2.1, 1.6, 2.5, 4.4, 0.0, 3.4, 2.9],
    [6.8, 5.9, 5.5, 7.1, 4.8, 3.3, 1.6, 6.9, 3.4, 0.0, 1.0],
    [6.0, 5.2, 4.8, 6.3, 4.1, 2.7, 0.9, 6.1, 2.9, 1.0, 0.0]
])

# 定义类别（红色点为0，黑色点为1）
labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])

# 计算错误分类的次数
error_count = 0


for i in range(len(labels)): #range(n) 函数生成从 0 到 n-1 的整数序列。

    # 获取当前样本的距离, 复制距离列表: 从距离矩阵中复制该样本与其他所有样本的距离 (distances = distance_matrix[i])。
    # distance_matrix[i] 这个表达式表示的是 distance_matrix 矩阵的第 i 行。
    distances = distance_matrix[i]
    
    # 将自身的距离设为无穷大，以排除自己
    distances[i] = np.inf
    
    # 找到最近邻
    nearest_neighbor = np.argmin(distances)
    #np.argmin 函数是 NumPy 库中的一个函数，它用于返回数组中最小值的索引——注意是索引，从0开始，而不是实际的数值。
    #如果不指定轴（axis），np.argmin 会返回将数组展平后最小值的索引。如果指定了轴，它会返回沿着该轴最小值的索引。
    
    # 检查最近邻的类别是否与当前样本的类别一致
    if labels[nearest_neighbor] != labels[i]:
        error_count += 1

# 计算错误率
error_rate = error_count / len(labels)
print(f"distances: {distances}")
print(f"nearest_neighbor: {nearest_neighbor}")
print(f"错误分类次数: {error_count}")
print(f"错误率: {error_rate:.2f}")

# 打印错误率（以分数形式表示）
print(f"错误率: {error_count}/{len(labels)}")
