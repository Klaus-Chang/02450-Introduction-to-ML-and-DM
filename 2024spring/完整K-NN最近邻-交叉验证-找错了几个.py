import numpy as np

# 欧几里得距离表（Table 5），使用NumPy数组表示
distance_matrix = np.array([
    [0, 0.2606, 1.1873, 2.4946, 2.9510, 2.5682, 3.4535, 2.4698],
    [0.2606, 0, 1.2796, 2.4442, 2.8878, 2.4932, 3.3895, 2.4216],
    [1.1873, 1.2796, 0, 2.8294, 3.6892, 2.9147, 4.1733, 2.2386],
    [2.4946, 2.4442, 2.8294, 0, 1.4852, 2.0608, 2.4941, 1.8926],
    [2.9510, 2.8878, 3.6892, 1.4852, 0, 1.5155, 1.0296, 1.3140],
    [2.5682, 2.4932, 2.9147, 2.0608, 1.5155, 0, 2.3316, 1.8870],
    [3.4535, 3.3895, 4.1733, 2.4941, 1.0296, 2.3316, 0, 0.7588],
    [2.4698, 2.4216, 2.2386, 1.8926, 1.3140, 1.8870, 0.7588, 0]
])

# 标签：0表示低油耗，1表示高油耗
labels = np.array([0, 0, 0, 0, 1, 1, 1, 0])

# 存储预测结果
predictions = np.zeros_like(labels)

# 执行1-NN分类
for i in range(len(labels)):
    # 获取距离矩阵中第i行的所有距离，排除自身
    distances = distance_matrix[i]
    distances[i] = np.inf  # 排除自己
    
    # 找到最近邻的索引
    nearest_neighbor_index = np.argmin(distances)
    
    # 最近邻的标签作为预测结果
    predictions[i] = labels[nearest_neighbor_index]

# 计算错误分类的个数
misclassified_count = np.sum(predictions != labels)

# 计算错误率
error_rate = misclassified_count / len(labels)

# 打印结果
print(f"预测结果: {predictions}")
print(f"实际标签: {labels}")
print(f"错误分类的个数: {misclassified_count}")
print(f"错误率: {error_rate:.4f}")
