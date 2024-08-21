import pandas as pd
from itertools import combinations

# 示例数据，实际数据应从OCR结果导入
data =[[1., 1., 1., 1., 0.],
[0., 0., 0., 0., 0.],
[1., 1., 0., 1., 0.],
[0., 1., 1., 0., 1.],
[1., 1., 1., 1., 1.],
[0., 0., 0., 0., 0.],
[1., 1., 0., 1., 0.],
[0., 1., 1., 0., 1.],
[1., 1., 1., 1., 0.],
[0., 1., 1., 0., 1.],
[0., 0., 0., 0., 0.],
[1., 1., 0., 1., 0.],
[0., 1., 1., 0., 1.],
[0., 1., 1., 0., 1.]
]

# 将数据转换为DataFrame
columns = ['x1', 'x2', 'x3', 'x4', 'x5']  # 根据实际OCR结果调整列名
df = pd.DataFrame(data, columns=columns)

# 确定要考虑的最大项集大小，这里假设为3
max_size = 3

# 总事务数
total_transactions = len(df)

# 存储支持度结果
support_dict = {}

# 生成并计算各项集的支持度
for size in range(1, max_size + 1):
    for subset in combinations(columns, size):
        # 计算所有选中列的交集（即这些项同时出现的次数）
        support_count = df[list(subset)].all(axis=1).sum()
        support = support_count / total_transactions
        support_dict[subset] = support

# 输出支持度大于特定阈值的项集，比如40%
threshold = 0.4
for itemset, supp in support_dict.items():
    if supp > threshold:
        print(f"Itemset {itemset} has a support of {supp:.2%}")
