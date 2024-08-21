import numpy as np
import matplotlib.pyplot as plt

def kmeans_convergence(data, initial_centroids, max_iterations=100, tol=1e-4):
    data = np.array(data)
    centroids = np.array(initial_centroids)
    K = len(centroids)
    
    for _ in range(max_iterations):
        # 分配数据点到最近的质心
        distances = np.abs(data[:, np.newaxis] - centroids)
        labels = np.argmin(distances, axis=1)
        
        # 更新质心位置
        new_centroids = np.array([data[labels == i].mean() if np.any(labels == i) else centroids[i] for i in range(K)])
        
        # 检查是否收敛
        if np.all(np.abs(new_centroids - centroids) < tol):
            break
        
        centroids = new_centroids
    
    return centroids, labels

def plot_clusters(data, labels, centroids):
    plt.figure(figsize=(10, 6))
    for i in range(len(centroids)):
        cluster_points = data[labels == i]
        plt.scatter(cluster_points, [i] * len(cluster_points), label=f'Cluster {i+1}')
    plt.scatter(centroids, range(len(centroids)), color='red', marker='x', s=200, linewidths=3, label='Centroids')
    plt.yticks(range(len(centroids)), [f'Cluster {i+1}' for i in range(len(centroids))])
    plt.xlabel('Data points')
    plt.title('K-means Clustering Result')
    plt.legend()
    plt.show()

# 主程序
if __name__ == "__main__":
    # 输入数据
    data = [-2.1,-1.7,-1.5,-0.4,0.0,0.6,0.8,1.0,1.1]
    
    # 设置初始质心位置（可以根据需要修改）
    initial_centroids = [-2.1,-1.7,-1.5]
    
    # 质心数量由初始质心的数量决定
    K = len(initial_centroids)
    
    # 执行K-means直到收敛
    final_centroids, labels = kmeans_convergence(data, initial_centroids)

    # 打印结果
    print(f"K = {K}")
    print("初始质心位置：", initial_centroids)
    print("\n最终的质心位置：")
    for i, centroid in enumerate(final_centroids):
        print(f"μ{i+1} = {centroid:.2f}")

    # 打印每个簇的成员
    clusters = [[] for _ in range(K)]
    for i, label in enumerate(labels):
        clusters[label].append(data[i])

    print("\n最终的簇：")
    for i, cluster in enumerate(clusters):
        print(f"簇 {i+1}: {sorted(cluster)}")

    # 可视化结果
    plot_clusters(np.array(data), labels, final_centroids)