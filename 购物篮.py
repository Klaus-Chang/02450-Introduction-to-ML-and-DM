import pandas as pd

# Construct the DataFrame from the user-uploaded data
data = pd.DataFrame({
    'Bread': [0, 1, 1, 1, 0],
    'Milk': [1, 1, 1, 1, 0],
    'Diaper': [1, 1, 0, 0, 0],
    'Beer': [1, 0, 1, 0, 1]
})

# Calculate the support for individual items
support_individual = data.sum() / len(data)

# Calculate the support for all pairs of items
pairs = [(col1, col2) for col1 in data.columns for col2 in data.columns if col1 < col2]
support_pairs = {pair: (data[list(pair)].all(axis=1).mean()) for pair in pairs}

# Filter pairs with support greater than 50%
support_pairs = {pair: support for pair, support in support_pairs.items() if support > 0.5}

# Combine individual items and pairs with support greater than 50%
results = support_individual[support_individual > 0.5].index.tolist() + list(support_pairs.keys())

print(results)

#这段Python程序是用来执行关联规则学习中的一个步骤，即计算项集的支持度（support）。程序的主要目的是找出单个商品和商品对的支持度，并筛选出支持度超过50%的项集。以下是程序的逐步解释：

#构建数据框架：
#使用pandas库创建一个DataFrame，其中包含了不同商品的购买情况。每一列代表一种商品，每一行代表一个事务（例如，一次购物篮的内容）。
#计算单个商品的支持度：
#对DataFrame中的每一列（商品）求和，然后除以总事务数（行数），得到每个商品的支持度。
#计算商品对的支持度：
##创建所有可能的商品对组合。
#对于每一对商品，计算同时购买这两种商品的事务比例，即这对商品的支持度。
#筛选支持度超过50%的商品对：
#检查所有商品对的支持度，只保留那些支持度超过50%的商品对。
#合并结果：
#将支持度超过50%的单个商品和商品对合并为一个列表。
#打印结果：
#输出支持度超过50%的单个商品和商品对。
#这个程序可以用于市场篮分析，帮助商家了解哪些商品经常一起被购买。这些信息可以用于商品布局、促销活动等。🛒

#例如，如果Milk和Bread的支持度超过50%，这意味着在超过一半的事务中，顾客同时购买了牛奶和面包。商家可能会考虑将这两种商品放在一起，或者在一种商品上做促销时也推荐另一种商品。