# 定义各个参数
K1 = 4  # 外层交叉验证的折数
K2 = 7  # 内层交叉验证的折数
model_variants = 3  # 模型架构的数量
training_time_per_model = 20  # 每次训练的时间（秒）
testing_time_per_model = 1    # 每次测试的时间（秒）

# 计算内层交叉验证的总时间
inner_fold_time = (training_time_per_model + testing_time_per_model) * K2

# 对每个模型架构进行内层交叉验证
total_inner_time_per_outer_fold = inner_fold_time * model_variants

# 外层训练和测试时间（最优模型）
outer_train_test_time = training_time_per_model + testing_time_per_model

# 计算总时间
total_time = (total_inner_time_per_outer_fold + outer_train_test_time) * K1

# 输出结果
print(f"The total time required for the 2-level cross-validation procedure is {total_time} seconds.")