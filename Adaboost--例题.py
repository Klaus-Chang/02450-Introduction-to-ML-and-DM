# 导入numpy库以进行数学运算
import numpy as np

# 假设我们有6个样本，其中包括正类和负类
samples = 6

# 初始权重，每个样本的权重相等
initial_weights = np.ones(samples) / samples

# 第一个分类器的错误率，基于误分类的样本计算
# 根据您的描述，有2个样本被误分类
epsilon1 = sum(initial_weights[:2])

# 使用公式计算alpha1: alpha1 = 0.5 * log((1 - epsilon1) / epsilon1)
alpha1 = 0.5 * np.log((1 - epsilon1) / epsilon1)

# 更新权重，对于正确分类的样本，权重乘以exp(-alpha)，对于误分类的样本，权重乘以exp(alpha)
updated_weights = np.array([weight * np.exp(-alpha1) if i >= 2 else weight * np.exp(alpha1)
                            for i, weight in enumerate(initial_weights)])

# 归一化权重，使它们的总和为1
updated_weights /= sum(updated_weights)

# 分别打印被分类错误的权重和分类正确的权重
misclassified_weights = updated_weights[:2]
correctly_classified_weights = updated_weights[2:]

print("被分类错误的权重:", misclassified_weights)
print("分类正确的权重:", correctly_classified_weights)
