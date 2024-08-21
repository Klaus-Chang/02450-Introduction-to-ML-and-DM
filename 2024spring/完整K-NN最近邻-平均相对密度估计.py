import numpy as np
from scipy.spatial.distance import pdist, squareform

def calculate_distance(x, y, distance_type='euclidean'):
    if distance_type == 'euclidean':
        return np.sqrt(np.sum((x - y)**2))
    elif distance_type == 'manhattan':
        return np.sum(np.abs(x - y))
    elif distance_type == 'cosine':
        return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    else:
        raise ValueError("Unsupported distance type")

def knn_density(data, k, target_index, distance_type='euclidean'):
    if isinstance(data, list):
        data = np.array(data)
    
    if data.ndim == 1:
        distances = pdist(data.reshape(-1, 1), metric=distance_type)
        distance_matrix = squareform(distances)
    elif data.ndim == 2:
        if data.shape[0] == data.shape[1]:  # 已经是距离矩阵
            distance_matrix = data
        else:  # 特征矩阵
            distances = pdist(data, metric=distance_type)
            distance_matrix = squareform(distances)
    else:
        raise ValueError("Invalid data format")
    
    distances = distance_matrix[target_index]
    distances[target_index] = np.inf  # 排除自身
    k_nearest = np.partition(distances, k)[:k]
    return 1 / np.mean(k_nearest)

def average_relative_density(data, k, target_index, distance_type='euclidean', density_type='knn'):
    if isinstance(data, list):
        data = np.array(data)
    
    if data.ndim == 1:
        distances = pdist(data.reshape(-1, 1), metric=distance_type)
        distance_matrix = squareform(distances)
    elif data.ndim == 2:
        if data.shape[0] == data.shape[1]:  # 已经是距离矩阵
            distance_matrix = data
        else:  # 特征矩阵
            distances = pdist(data, metric=distance_type)
            distance_matrix = squareform(distances)
    else:
        raise ValueError("Invalid data format")
    
    target_density = knn_density(distance_matrix, k, target_index, distance_type)
    
    distances = distance_matrix[target_index]
    distances[target_index] = np.inf  # 排除自身
    k_nearest_indices = np.argpartition(distances, k)[:k]
    
    neighbor_densities = [knn_density(distance_matrix, k, i, distance_type) for i in k_nearest_indices]
    
    ard = target_density / np.mean(neighbor_densities)
    return ard

# 使用示例
if __name__ == "__main__":
    # 数据可以是距离矩阵或特征矩阵
    data = np.array([
    [0., 0.534, 1.257, 1.671, 1.09, 1.315, 1.484, 1.253, 1.418],
    [0.534, 0., 0.727, 2.119, 1.526, 1.689, 1.214, 0.997, 1.056],
    [1.257, 0.727, 0., 2.809, 2.22, 2.342, 1.088, 0.965, 0.807],
    [1.671, 2.119, 2.809, 0., 0.601, 0.54, 3.135, 2.908, 3.087],
    [1.09, 1.526, 2.22, 0.601, 0., 0.331, 2.563, 2.338, 2.5],
    [1.315, 1.689, 2.342, 0.54, 0.331, 0., 2.797, 2.567, 2.708],
    [1.484, 1.214, 1.088, 3.135, 2.563, 2.797, 0., 0.275, 0.298],
    [1.253, 0.997, 0.965, 2.908, 2.338, 2.567, 0.275, 0., 0.343],
    [1.418, 1.056, 0.807, 3.087, 2.5, 2.708, 0.298, 0.343, 0.]
    ])

    target_index = 3  # 观察点的索引（例如，o4）
    k = 1  # K最近邻的K值
    distance_type = 'euclidean'  # 可以选择 'euclidean', 'manhattan', 'cosine' 等

    ard = average_relative_density(data, k, target_index, distance_type)
    print(f"观察点 {target_index + 1} 的平均相对密度 (K={k}): {ard:.4f}")

    # 检查哪个选项最接近
    options = [1.0, 0.71, 0.68, 0.36]
    closest_option = min(options, key=lambda x: abs(x - ard))
    print(f"最接近的选项: {closest_option}")