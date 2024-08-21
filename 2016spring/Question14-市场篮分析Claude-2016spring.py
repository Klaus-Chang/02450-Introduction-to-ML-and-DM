from itertools import combinations

def calculate_support(data):
    n_transactions = len(data)
    n_items = len(data[0])
    
    all_itemsets = []
    for i in range(1, 4):  # 1到3项的组合
        all_itemsets.extend(combinations(range(n_items), i))
    
    support_dict = {}
    for itemset in all_itemsets:
        count = sum(all(row[item] == 1 for item in itemset) for row in data)
        support = count / n_transactions
        if count >= 6:  # 直接判断出现次数，避免浮点数比较
            support_dict[itemset] = support
    
    return support_dict

# 使用您提供的数据
data = [[1., 1., 1., 1., 0.],
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

support_dict = calculate_support(data)

print("支持度大于等于 6/14 的项集：")
for itemset, support in support_dict.items():
    print(f"{[f'x{i+1}' for i in itemset]}: {support:.2%}")