import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 预设的数据输入
def get_preset_data():
    # 这里可以放置您的预设数据
    # 例如，一个简单的2D数据集
    X = np.array([
        [1, 2],
        [1.5, 1.8],
        [5, 8],
        [8, 8],
        [1, 0.6],
        [9, 11],
        [8, 2],
        [10, 2],
        [9, 3]
    ])
    K = 3  # 预设的聚类数
    return X, K

    # 如果您想使用3D数据，可以使用下面的示例：
    # X = np.array([
    #     [1, 2, 3],
    #     [1.5, 1.8, 2],
    #     [5, 8, 4],
    #     [8, 8, 7],
    #     [1, 0.6, 9],
    #     [9, 11, 3],
    #     [8, 2, 5],
    #     [10, 2, 8],
    #     [9, 3, 6]
    # ])
    # K = 3
    # return X, K

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

def kmeans(X, K, max_iters=100):
    # 随机初始化质心
    centroids = X[np.random.choice(X.shape[0], K, replace=False)]
    
    for _ in range(max_iters):
        # 为每个点分配最近的质心
        labels = np.zeros(X.shape[0], dtype=int)
        for i, point in enumerate(X):
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            labels[i] = np.argmin(distances)
        
        # 更新质心
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])
        
        # 检查收敛
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return labels, centroids

def plot_clusters(X, labels, centroids):
    dim = X.shape[1]
    if dim == 2:
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidths=3)
        plt.title('2D K-Means Clustering')
        plt.show()
    elif dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis')
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='red', marker='x', s=200, linewidths=3)
        ax.set_title('3D K-Means Clustering')
        plt.show()
    else:
        print("Can only plot 2D or 3D data")

# 主程序
if __name__ == "__main__":
    X, K = get_preset_data()
    labels, centroids = kmeans(X, K)
    plot_clusters(X, labels, centroids)
    
    print("\nFinal Centroids:")
    for i, centroid in enumerate(centroids):
        print(f"Centroid {i+1}: {centroid}")

    # 打印每个点的标签
    print("\nPoint Labels:")
    for i, label in enumerate(labels):
        print(f"Point {i+1}: Cluster {label+1}")