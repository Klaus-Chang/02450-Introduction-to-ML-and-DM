import numpy as np
from scipy.spatial.distance import pdist, squareform

def knn_density(distance_matrix, k, exclude_self=True):
    """
    计算每个点的K-最近邻密度
    
    参数:
    distance_matrix: 距离矩阵
    k: 最近邻的数量
    exclude_self: 是否在计算中排除自身
    
    返回:
    每个点的KNN密度
    """
    n = distance_matrix.shape[0]
    densities = np.zeros(n)
    
    for i in range(n):
        distances = distance_matrix[i]
        if exclude_self:
            distances[i] = np.inf  # 排除自身
        
        # 找到K个最近邻
        k_nearest = np.sort(distances)[:k]
        
        # 计算密度 (使用K-距离的倒数)
        densities[i] = 1 / np.mean(k_nearest)
    
    return densities

# 输入距离矩阵
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

# 计算o1的3-最近邻密度
k = 3
densities = knn_density(distance_matrix, k)
o1_density = densities[1]

print(f"o1的{k}-最近邻密度: {o1_density:.3f}")

# 检查哪个选项最接近
options = [0.625, 0.462, 1.139, 0.526]
closest_option = min(options, key=lambda x: abs(x - o1_density))
print(f"最接近的选项: {closest_option}")