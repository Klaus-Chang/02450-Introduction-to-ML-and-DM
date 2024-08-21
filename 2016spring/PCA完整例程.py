import numpy as np
from sklearn.preprocessing import StandardScaler

# 原始数据矩阵
X = np.array([[2.5, 2.4],
              [0.5, 0.7],
              [2.2, 2.9],
              [1.9, 2.2],
              [3.1, 3.0],
              [2.3, 2.7],
              [2.0, 1.6],
              [1.0, 1.1],
              [1.5, 1.6],
              [1.1, 0.9]])

# 步骤1：数据标准化
scaler = StandardScaler()
Z = scaler.fit_transform(X)

# 步骤2：计算协方差矩阵
C = np.cov(Z, rowvar=False)

# 步骤3：计算协方差矩阵的特征值和特征向量（使用SVD）
U, Sigma, VT = np.linalg.svd(Z)
V = VT.T
# SVD分解得到对角矩阵
Sigma_matrix = np.diag(Sigma)

# 协方差矩阵的特征值为奇异值的平方除以样本量减1
eigenvalues = Sigma**2 / (Z.shape[0] - 1)
# 特征向量为V的列向量
eigenvectors = V

# 打印特征值和特征向量
print("特征值:\n", eigenvalues)
print("特征向量:\n", eigenvectors)

# 步骤4：选择主要成分（假设我们选择第一个主要成分）
P = eigenvectors[:, 0].reshape(-1, 1)

# 步骤5：将原始数据投影到主成分上
Y = np.dot(Z, P)

# 打印降维后的数据
print("降维后的数据:\n", Y)
