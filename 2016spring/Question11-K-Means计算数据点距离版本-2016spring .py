import numpy as np

# 数据点和初始质心
points = np.array([-2.1, -1.7, -1.5, -0.4, 0.0, 0.6, 0.8, 1.0, 1.1])
centroids = np.array([-2.1, -1.7, -1.5])  # 初始为O4, O6, O5

# 对应观测点标签
labels = ['O4', 'O6', 'O5', 'O1', 'O2', 'O3', 'O8', 'O9', 'O7']

def kmeans(points, centroids):
    iteration = 0
    while True:
        print(f"\nIteration {iteration + 1}:")
        # 计算每个点到各个质心的距离
        distances = np.abs(points[:, np.newaxis] - centroids)
        # 为每个点分配最近的质心
        closest_centroid_indices = np.argmin(distances, axis=1)
        
        # 基于当前分配，计算新质心
        new_centroids = np.array([points[closest_centroid_indices == k].mean() for k in range(len(centroids))])
        
        print("New centroids:", new_centroids)
        print("Distances to new centroids:")
        for point, label in zip(points, labels):
            dist_to_new_centroids = np.abs(point - new_centroids)
            print(f"{label}: {dict(zip(centroids, dist_to_new_centroids))}")
        
        # 检查质心是否改变
        if np.all(new_centroids == centroids):
            break
        
        centroids = new_centroids
        iteration += 1
        
    return centroids, closest_centroid_indices

# 执行K-means算法
final_centroids, assignments = kmeans(points, centroids)
