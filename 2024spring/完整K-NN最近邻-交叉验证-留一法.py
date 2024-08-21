import numpy as np
from collections import Counter

def knn_classifier(distances, labels, k, test_index):
    test_distances = distances[test_index]
    mask = np.ones(len(labels), dtype=bool)
    mask[test_index] = False
    
    # 获取所有其他样本的距离和标签
    other_distances = test_distances[mask]
    other_labels = labels[mask]
    
    # 找到k个最近邻
    nearest_indices = np.argsort(other_distances)[:k]
    nearest_distances = other_distances[nearest_indices]
    nearest_labels = other_labels[nearest_indices]
    
    # 统计最近邻中各类别的数量
    label_counts = Counter(nearest_labels)
    max_count = max(label_counts.values())
    
    # 如果存在平局
    if list(label_counts.values()).count(max_count) > 1:
        # 找出票数最高的类别
        top_labels = [label for label, count in label_counts.items() if count == max_count]
        # 在这些类别中选择距离最近的
        for i in range(k):
            if nearest_labels[i] in top_labels:
                return nearest_labels[i]
    else:
        return label_counts.most_common(1)[0][0]

def loocv_knn(distances, labels, k):
    n = len(labels)
    predictions = []
    
    for i in range(n):
        pred = knn_classifier(distances, labels, k, i)
        predictions.append(pred)
    
    error_rate = np.mean(np.array(predictions) != labels)
    return error_rate

# 输入数据
distances = np.array([
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

labels = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2])

# 执行3-最近邻LOOCV
k = 1
error_rate = loocv_knn(distances, labels, k)

print(f"Error rate for {k}-NN using LOOCV: {error_rate:.2f}")

# 检查哪个选项最接近
options = [1/5, 5/10, 6/10, 7/10]
closest_option = min(options, key=lambda x: abs(x - error_rate))
print(f"Closest option: {closest_option}")