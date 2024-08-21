import numpy as np


# 假设Z和Q是包含聚类标签的列表或数组
Z = [1, 1, 1, 1, 2, 2, 3, 3, 3]  # 示例中的聚类结果Z
Q = [4, 4, 1, 1, 2, 2, 2, 3, 3]  # 示例中的聚类结果Q

# 确定聚类数量: 首先，代码通过 set(Z) 和 set(Q) 确定了Z和Q中的唯一聚类标签数量，并分别存储在 num_clusters_Z 和 num_clusters_Q 中。
# 使用 set(Z) 和 set(Q) 来确定两个聚类结果中各自包含的独立聚类的数量。这是因为集合（set）只会包含唯一元素，长度（使用 len() 函数）将给出不同聚类的数目。
num_clusters_Z = len(set(Z))
num_clusters_Q = len(set(Q))

# 创建一个空的n矩阵
# n 矩阵的大小是 num_clusters_Z x num_clusters_Q。num_clusters_Z 是聚类结果Z 中的聚类数量，num_clusters_Q 是聚类结果 Q 中的聚类数量。
# 初始化 n 矩阵为零矩阵，表示开始时没有任何观察值被记录在交叉聚类对中。
n = np.zeros((num_clusters_Z, num_clusters_Q), dtype=int)

# 填充n矩阵
# 遍历每个观察值的聚类标签，对于聚类结果 Z 和 Q 中的每个观察值，递增相应的 n 矩阵的元素。
# zip(Z, Q) 会生成一个元组列表，其中每个元组包含同一观察值在 Z 和 Q 中的聚类标签。
#对于每对 (z_label, q_label)，n[z_label-1, q_label-1] 的值增加 1，表示对应的聚类对中观察值的数量增加。这里的 -1 是因为 Python 的索引从 0 开始，而聚类标签通常从 1 开始。

for z_label, q_label in zip(Z, Q):
    n[z_label-1, q_label-1] += 1

print("n矩阵:\n", n)

# 假设n矩阵、nZ和nQ已经给出或已经计算得出
#n = np.array([[2, 0, 0, 2],
#              [0, 2, 0, 0],
#              [0, 1, 2, 0]])

nZ = np.sum(n, axis=1)#这行代码计算的是矩阵n的每一行的和，结果是一个数组，其中包含了每一行元素的总和。这里的axis=1表示沿着水平方向（行）进行操作。
nQ = np.sum(n, axis=0)#这行代码计算的是矩阵n的每一列的和，结果同样是一个数组，包含了每一列元素的总和。这里的axis=0表示沿着垂直方向（列）进行操作。

# 计算S（Agreement）
S = np.sum(n * (n - 1) / 2)

# 计算N（所有观察值的总数）
N = np.sum(nZ)

# 计算总对数
total_pairs = N * (N - 1) / 2

# 计算D（Disagreement）
D = total_pairs - np.sum(nZ * (nZ - 1) / 2) - np.sum(nQ * (nQ - 1) / 2) + S  # 修改为正确的计算方式


print("n矩阵:\n", n)
print("nZ:", nZ)
print("nQ:", nQ)
print("同一聚类中的观察值对数S（Agreement）:", S)
print("不同聚类中的观察值对数D（Disagreement）:", D)