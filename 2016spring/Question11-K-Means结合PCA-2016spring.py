import numpy as np
import matplotlib.pyplot as plt

# 数据点
points = np.array([-2.1, -1.7, -1.5, -0.4, 0.0, 0.6, 0.8, 1.0, 1.1])
# 初始质心
centroids = np.array([-2.1, -1.7, -1.5])

def k_means_1d(points, initial_centroids):
    centroids = initial_centroids.copy()
    prev_centroids = centroids.copy()
    while True:
        # 分配数据点到最近的质心
        clusters = {}
        for point in points:
            distances = np.abs(point - centroids)
            closest_centroid = np.argmin(distances)
            if closest_centroid not in clusters:
                clusters[closest_centroid] = []
            clusters[closest_centroid].append(point)
        
        # 更新质心
        for key in clusters.keys():
            centroids[key] = np.mean(clusters[key])
        
        # 检查质心是否收敛
        if np.all(prev_centroids == centroids):
            break
        
        prev_centroids = centroids.copy()
    
    return centroids, clusters

# 运行K-means算法
final_centroids, final_clusters = k_means_1d(points, centroids)

# 打印结果
print("最终质心位置：", final_centroids)
print("最终簇分配：", final_clusters)

# 可视化结果
colors = ['r', 'g', 'b']
for key in final_clusters.keys():
    plt.scatter(final_clusters[key], [key]*len(final_clusters[key]), color=colors[key])
plt.scatter(final_centroids, [0, 1, 2], color='k', marker='x', s=100, label='Centroids')
plt.xlabel('PCA1')
plt.title('K-means Clustering')
plt.legend()
plt.show()
