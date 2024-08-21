import numpy as np

# 模型系数和截距
coefficients = np.array([0.76, 1.76, -0.32, -0.96, 6.64, -5.13, -2.06, 96.73, 1.03, -2.74])
intercept = 1.41

# x6的特征，含地区的独热编码
x6_features = np.array([-0.06, -0.28, 0.43, -0.30, -0.36, 0, 0, 0, 0, 1])

# 计算逻辑回归的线性组合部分
linear_combination = np.dot(coefficients, x6_features) + intercept

# 使用sigmoid函数计算预测概率
probability = 1 / (1 + np.exp(-linear_combination))

print(f"预测的类别1的概率是: {probability:.4f}")