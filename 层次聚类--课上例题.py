import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

#导入必要的库：

#numpy用于数据处理。
##scipy.cluster.hierarchy的linkage用于执行层次聚类，dendrogram用于绘制树状图。
#matplotlib.pyplot用于绘图。


# 数据点
data = np.array([42, 60, 17, 48, 12])

# 将数据转换为2D数组，因为linkage函数需要2D数组输入
data = data[:, np.newaxis]

# 使用linkage函数，指定使用最小链接法（'single'）和欧几里得距离（'euclidean'）来聚合聚类。
Z = linkage(data, method='single', metric='euclidean')

# 绘制树状图，使用dendrogram函数显示聚类结果，指定数据点的标签以提高图表的可读性。
plt.figure(figsize=(10, 8))
dendrogram(Z, labels=['42', '60', '17', '48', '12'])
plt.title("Dendrogram for the dataset using Single Linkage")
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distance')
plt.show()
