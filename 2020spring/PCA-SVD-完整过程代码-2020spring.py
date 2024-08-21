import numpy as np
from sklearn.preprocessing import StandardScaler

# Step 1: 生成假设的数据集并标准化
def generate_and_standardize_data():
    # 假设的数据集
    data = np.array([[2.5, 2.4],
                     [0.5, 0.7],
                     [2.2, 2.9],
                     [1.9, 2.2],
                     [3.1, 3.0],
                     [2.3, 2.7],
                     [2.0, 1.6],
                     [1.0, 1.1],
                     [1.5, 1.6],
                     [1.1, 0.9]])

    # 标准化数据集
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)
    return standardized_data

# 计算对角矩阵（奇异值矩阵）
def compute_singular_value_matrix(data):
    U, S, VT = np.linalg.svd(data, full_matrices=False)
    return S

# Step 2: 计算特征值（奇异值的平方除以样本数减一）
def compute_eigenvalues(singular_values, n_samples):
    return np.square(singular_values) / (n_samples - 1)

# Step 3: 计算解释方差
def compute_explained_variance(eigenvalues):
    total_variance = np.sum(eigenvalues)
    explained_variance = eigenvalues / total_variance
    return explained_variance

# 计算累计解释方差
def compute_cumulative_explained_variance(explained_variance):
    cumulative_explained_variance = np.cumsum(explained_variance)
    return cumulative_explained_variance

# 主程序
standardized_data = generate_and_standardize_data()
singular_values = compute_singular_value_matrix(standardized_data)
print("Singular Values (σ):\n", singular_values)

n_samples = standardized_data.shape[0]
eigenvalues = compute_eigenvalues(singular_values, n_samples)
print("Eigenvalues:\n", eigenvalues)

explained_variance = compute_explained_variance(eigenvalues)
print("Explained Variance:\n", explained_variance)

cumulative_explained_variance = compute_cumulative_explained_variance(explained_variance)
print("Cumulative Explained Variance:\n", cumulative_explained_variance)

# 根据题目计算特定的解释方差
print("\nSpecific Principal Component Variance Calculations:")
print(f"The variance explained by the first four principal components: {cumulative_explained_variance[3] * 100:.2f}%")
print(f"The variance explained by the last four principal components: {sum(explained_variance[-4:]) * 100:.2f}%")
print(f"The variance explained by the first two principal components: {cumulative_explained_variance[1] * 100:.2f}%")
print(f"The variance explained by the first principal component: {explained_variance[0] * 100:.2f}%")
