import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier

# 输入距离矩阵
distance_matrix = np.array([
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

# 标签
labels = np.array(['Kama', 'Kama', 'Kama', 'Rosa', 'Rosa', 'Rosa', 'Canadian', 'Canadian', 'Canadian'])

# 创建留一法交叉验证器
loo = LeaveOneOut()
correct_classifications = 0

# 执行LOOCV
for train_index, test_index in loo.split(distance_matrix):
    # 创建一个基于预计算距离的2-NN分类器，使用距离加权
    classifier = KNeighborsClassifier(n_neighbors=2, metric='precomputed', weights='distance')
    #**weights** (默认为 ‘uniform’):
    #- ‘uniform’: 所有邻居的权重都相同。
    #- ‘distance’: 权重与距离成反比，即更近的邻居对分类结果的影响更大。
    
    # 训练模型
    classifier.fit(distance_matrix[train_index][:, train_index], labels[train_index])
    
    # 预测
    prediction = classifier.predict(distance_matrix[test_index][:, train_index])
    
    # 检查预测是否正确
    if prediction == labels[test_index]:
        correct_classifications += 1

# 计算精确度
accuracy = correct_classifications / len(labels)
print(f'Accuracy of 2-NN classifier using LOOCV: {accuracy:.2f}')
