import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram #用于层次聚类。

# 定义正确的距离矩阵
distance_matrix = np.array([
[0., 10.4, 11.3, 3.5, 10.3, 6.3, 5.1, 8.6, 10.8, 11.],
[10.4, 0., 5.5, 7.1, 0.4, 4.1, 5.3, 1.8, 4.4, 6.1],
[11.3, 5.5, 0., 8., 5.8, 9., 10.2, 6.7, 4.6, 6.4],
[3.5, 7.1, 8., 0., 7., 3.7, 2.7, 5.4, 7.5, 8.2],
[10.3, 0.4, 5.8, 7., 0., 4., 5.2, 1.7, 4.5, 5.9],
[6.3, 4.1, 9., 3.7, 4., 0., 1.7, 2.3, 8.5, 9.1],
[5.1, 5.3, 10.2, 2.7, 5.2, 1.7, 0., 3.5, 9.7, 9.9],
[8.6, 1.8, 6.7, 5.4, 1.7, 2.3, 3.5, 0., 6.2, 6.8],
[10.8, 4.4, 4.6, 7.5, 4.5, 8.5, 9.7, 6.2, 0., 2.5],
[11., 6.1, 6.4, 8.2, 5.9, 9.1, 9.9, 6.8, 2.5, 0.]
])

# 将距离矩阵转换为压缩格式（scipy需要这种格式）
condensed_distance_matrix = distance_matrix[np.triu_indices(10, k=1)]

# 使用最大链式法进行层次聚类
# 在层次聚类中，linkage 函数的 method 参数除了 'complete' 最大链式法，还包括以下几种可选参数：

# 'single'：最近邻法，也称为最小链式法，它考虑的是簇间最近的两个样本的距离。
# 'average'：平均链式法，计算所有可能配对的样本之间的平均距离。
# 'weighted'：加权平均法，考虑到簇大小在计算平均距离时的权重。
# 'centroid'：质心法，考虑两个簇的质心（中心点）之间的距离。
# 'median'：中位数法，类似于质心法，但使用中位数来计算簇的中心。
# 'ward'：沃德法，基于最小化簇内平方和的原则来合并簇

Z = linkage(condensed_distance_matrix, method='centroid')

# 绘制树状图
plt.figure(figsize=(10, 7))
dendrogram(Z, labels=[f'o{i}' for i in range(1, 11)])
plt.title('Dendrogram using Maximum Linkage')
plt.xlabel('Observations')
plt.ylabel('Distance')
plt.show()
