import numpy as np
from sklearn.preprocessing import StandardScaler

# 给定的奇异值对角元素
singular_values = np.array([16.4, 11.22, 9.1, 6.06, 0.03])

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


# 样本数量（这里假设为标准化后数据的样本数）
n_samples = 25  # 请根据实际数据集确定

# 计算特征值（奇异值的平方除以样本数减一）
eigenvalues = compute_eigenvalues(singular_values, n_samples)
print("Eigenvalues:\n", eigenvalues)

# 计算解释方差
explained_variance = compute_explained_variance(eigenvalues)
print("Explained Variance:\n", explained_variance)

# 计算累计解释方差
cumulative_explained_variance = compute_cumulative_explained_variance(explained_variance)
print("Cumulative Explained Variance:\n", cumulative_explained_variance)

# 根据题目计算特定的解释方差
print("\nSpecific Principal Component Variance Calculations:")
print(f"The variance explained by the first four principal components: {cumulative_explained_variance[3] * 100:.2f}%")
print(f"The variance explained by the last four principal components: {sum(explained_variance[-4:]) * 100:.2f}%")
print(f"The variance explained by the first two principal components: {cumulative_explained_variance[1] * 100:.2f}%")
print(f"The variance explained by the first principal component: {explained_variance[0] * 100:.2f}%")
