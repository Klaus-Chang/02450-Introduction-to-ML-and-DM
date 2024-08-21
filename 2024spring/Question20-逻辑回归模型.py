import numpy as np

# 输入的特征向量 x3（已经标准化）
x3 = np.array([1.209, -0.303, -1.070, -0.459, 0.013])

# 模型 A 的权重向量
wA = np.array([-1.332, 0.378])
# 模型 B 的权重向量
wB = np.array([-1.346, 0.387, -0.222])
# 模型 C 的权重向量
wC = np.array([-15.913, 9.028, -0.273, 18.852, 2.133, -5.583])

# 计算模型 A 的预测值
zA = np.dot(wA, np.append(1, x3[0]))  # 只使用 x1
yA = 1 / (1 + np.exp(-zA))

# 计算模型 B 的预测值
zB = np.dot(wB, np.append(1, x3[:2]))  # 使用 x1 和 x2
yB = 1 / (1 + np.exp(-zB))

# 计算模型 C 的预测值
zC = np.dot(wC, np.append(1, x3))  # 使用全部特征
yC = 1 / (1 + np.exp(-zC))

print(f"\nzA, yA, zB, yB, zC, yC: {zA, yA, zB, yB, zC, yC}")