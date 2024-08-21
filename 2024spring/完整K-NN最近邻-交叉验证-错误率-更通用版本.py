import numpy as np
from scipy.spatial.distance import pdist, squareform
from collections import Counter
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split

def calculate_distance(x, y, distance_type='cityblock'):
    if distance_type == 'cityblock':
        return np.sum(np.abs(x - y))
    elif distance_type == 'euclidean':
        return np.sqrt(np.sum((x - y)**2))
    elif distance_type == 'cosine':
        return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    else:
        raise ValueError("Unsupported distance type")

def knn_classifier(distances, labels, k, test_indices):
    predictions = []
    for test_index in test_indices:
        test_distances = distances[test_index]
        sorted_indices = np.argsort(test_distances)
        k_nearest_labels = labels[sorted_indices[1:k+1]]  # Exclude the point itself
        predictions.append(Counter(k_nearest_labels).most_common(1)[0][0])
    return predictions

def knn_density(distances, k, target_indices):
    densities = []
    for target_index in target_indices:
        target_distances = distances[target_index]
        target_distances[target_index] = np.inf  # Exclude self
        k_nearest = np.partition(target_distances, k)[:k]
        densities.append(1 / np.mean(k_nearest))
    return densities

def cross_validation(distances, labels, k, task, cv_method='loo', n_splits=5):
    if cv_method == 'loo':
        cv = LeaveOneOut()
    elif cv_method == 'kfold':
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        raise ValueError("Unsupported cross-validation method")

    results = []
    for train_index, test_index in cv.split(distances):
        if task == 'classification':
            pred = knn_classifier(distances, labels, k, test_index)
            results.extend(np.array(pred) != labels[test_index])
        elif task == 'density':
            densities = knn_density(distances, k, test_index)
            results.extend(densities)

    if task == 'classification':
        return np.mean(results)
    elif task == 'density':
        return results

def holdout_validation(distances, labels, k, task, test_size=0.2):
    train_index, test_index = train_test_split(range(len(labels)), test_size=test_size, random_state=42)
    
    if task == 'classification':
        pred = knn_classifier(distances, labels, k, test_index)
        error_rate = np.mean(np.array(pred) != labels[test_index])
        return error_rate
    elif task == 'density':
        densities = knn_density(distances, k, test_index)
        return densities

def main(data, labels=None, k=3, task='classification', distance_type='cityblock', cv_method='loo', n_splits=5):
    if isinstance(data, list):
        data = np.array(data)

    if data.ndim == 2 and data.shape[0] == data.shape[1]:
        distances = data  # Already a distance matrix
    else:
        distances = pdist(data, metric=distance_type)
        distances = squareform(distances)

    if task == 'classification':
        if labels is None:
            raise ValueError("Labels are required for classification task")
        error_rate = cross_validation(distances, np.array(labels), k, task, cv_method, n_splits)
        print(f"Error rate for {k}-NN using {cv_method.upper()}: {error_rate:.4f}")
        
        # Add holdout method
        holdout_error_rate = holdout_validation(distances, np.array(labels), k, task)
        print(f"Error rate for {k}-NN using Holdout method: {holdout_error_rate:.4f}")
        
        return error_rate, holdout_error_rate
    elif task == 'density':
        densities = cross_validation(distances, np.array(range(len(data))), k, task, cv_method, n_splits)
        return densities

# 使用示例
if __name__ == "__main__":
    # 数据（距离矩阵）
    data = np.array([
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

    # 类别标签
    labels = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2])

    # 分类任务 - 留一法交叉验证和留出法
    # 可以选择 'euclidean', 'manhattan', 'cosine' 等
    error_rate_loo, error_rate_holdout = main(data, labels, k=1, task='classification', distance_type='manhattan', cv_method='loo')

    # 分类任务 - K折交叉验证
    error_rate_kfold, _ = main(data, labels, k=3, task='classification', distance_type='euclidean', cv_method='kfold', n_splits=5)

    # 密度估计任务
    densities = main(data, k=3, task='density', distance_type='euclidean', cv_method='loo')

    print(f"\nDensity for observation 1: {densities[0]:.4f}")

    # 检查哪个选项最接近（针对分类任务 - 留一法）
    options = [3/10, 5/10, 6/10, 7/10]
    closest_option_loo = min(options, key=lambda x: abs(x - error_rate_loo))
    print(f"Closest option for classification error rate (LOO): {closest_option_loo}")

    # 检查哪个选项最接近（针对分类任务 - K折）
    closest_option_kfold = min(options, key=lambda x: abs(x - error_rate_kfold))
    print(f"Closest option for classification error rate (K-Fold): {closest_option_kfold}")

    # 检查哪个选项最接近（针对分类任务 - 留出法）
    closest_option_holdout = min(options, key=lambda x: abs(x - error_rate_holdout))
    print(f"Closest option for classification error rate (Holdout): {closest_option_holdout}")

    # 检查哪个选项最接近（针对密度估计任务，以第一个观察为例）
    density_options = [0.625, 0.462, 1.139, 0.526]
    closest_density_option = min(density_options, key=lambda x: abs(x - densities[0]))
    print(f"Closest option for density of observation 1: {closest_density_option}")