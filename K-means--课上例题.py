from sklearn.cluster import KMeans
import numpy as np

# 数据集
X = np.array([[42], [60], [17], [48], [12]])

# 初始化KMeans
kmeans = KMeans(n_clusters=2, init=np.array([[17], [12]]), n_init=1)

# 拟合模型
kmeans.fit(X)

# 聚类中心
centers = kmeans.cluster_centers_

# 输出每个聚类中的数据点
clusters = {}
for i in range(len(X)):
    label = kmeans.labels_[i]
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(X[i][0])

# 打印结果
for label, points in clusters.items():
    print(f"聚类 {label} 包含的数据点: {points}")
print(f"聚类中心: {centers}")

#n_clusters=2 表示我们想要形成的聚类数量。
#init 参数用于指定初始的聚类中心，这里我们使用了您提供的初始值。
#_init=1 表示算法运行一次，使用指定的初始中心。
#这段代码首先使用KMeans类对数据集进行聚类，然后创建一个字典clusters来存储每个聚类中的数据点。最后，它遍历每个聚类并打印出包含的数据点和聚类中心。
#运行这段代码，您将能看到每个聚类中包含哪些数据点，以及最终的聚类中心。这样您就可以直观地了解到数据是如何被分组的