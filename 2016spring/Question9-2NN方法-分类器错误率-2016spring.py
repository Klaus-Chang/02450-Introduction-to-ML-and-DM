import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier

#根据题目的说明，特别是关于如何处理平局情况（tied classes），你的代码需要相应的调整以确保正确处理最近邻之间的平局。
#目前的实现使用 KNeighborsClassifier ，它会自动进行多数投票来决定分类。如果出现平局，这个类库默认会选择标签排序中较小的那一个，而不是基于距离选择最近的邻居。
#为了符合题目的要求，你可以手动实现2-NN的逻辑，以确保在出现平局时，根据最近的邻居来分类。以下是如何修改你的代码的建议：

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

# 用于跟踪正确分类的次数
#correct_classifications = 0

# 执行LOOCV
#for train_index, test_index in loo.split(distance_matrix):
    # 创建一个基于预计算距离的2-NN分类器
    #classifier = KNeighborsClassifier(n_neighbors=2, metric='precomputed')
    
    # 训练模型
    #classifier.fit(distance_matrix[train_index][:, train_index], labels[train_index])
    
    # 预测
    #prediction = classifier.predict(distance_matrix[test_index][:, train_index])
    
    # 检查预测是否正确
    #if prediction == labels[test_index]:
        #correct_classifications += 1

# 计算精确度
#accuracy = correct_classifications / len(labels)
#print(f'Accuracy of 2-NN classifier using LOOCV: {accuracy:.2f}')

for train_index, test_index in loo.split(distance_matrix):
    train_distances = distance_matrix[test_index][:, train_index]
    nearest_indices = np.argsort(train_distances)[0, :2]  # 获取两个最近邻的索引
    nearest_labels = labels[train_index][nearest_indices]  # 获取这两个索引对应的标签
    
    # 检查是否平局
    if nearest_labels[0] == nearest_labels[1]:
        prediction = nearest_labels[0]  # 如果两个最近邻标签相同，直接预测
    else:
        # 如果标签不同，则选择最近的邻居的标签
        prediction = labels[train_index][nearest_indices[0]]

    if prediction == labels[test_index]:
        correct_classifications += 1

accuracy = correct_classifications / len(labels)
print(f'Accuracy of 2-NN classifier using LOOCV: {accuracy:.2f}')