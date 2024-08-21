import pandas as pd

# 表4的数据
data = {
    'features': [
        'none', 'x1', 'x2', 'x3', 'x4', 
        'x1, x2', 'x1, x3', 'x1, x4', 'x2, x3', 'x2, x4', 
        'x3, x4', 'x1, x2, x3', 'x1, x2, x4', 
        'x1, x3, x4', 'x2, x3, x4', 'x1, x2, x3, x4',
    ],
    'Training RMSE': [0.6667, 0.1143, 0.1143, 0.219, 0.1524, 0.0952, 0.1143, 0.1143, 0.1238, 0.1048, 0.1143, 0.0571, 0.1048, 0.0857, 0.0762, 0.0667],
    'Test RMSE': [0.6667, 0.1524, 0.1143, 0.1714, 0.1714, 0.1619, 0.1619, 0.1619, 0.1333, 0.1429, 0.1619, 0.1714, 0.1619, 0.1619, 0.1524, 0.181]
}

# 检查数据长度是否一致
print(len(data['features']), len(data['Training RMSE']), len(data['Test RMSE']))

df = pd.DataFrame(data)

def forward_selection(df):
    """
    前向选择算法：每一步都选择一个新的特征添加到模型中，使得模型的测试RMSE最小。
    参数：
        df (DataFrame)：包含特征组合及其对应的训练和测试RMSE的数据框。
    返回：
        best_features (list)：选择的最佳特征组合列表。
    """
    remaining_features = set(['x1', 'x2', 'x3', 'x4'])
    selected_features = set()
    current_rmse = float('inf')
    best_features = []
    
    while remaining_features:
        best_rmse = current_rmse
        best_feature = None
        
        for feature in remaining_features:
            candidate_features = selected_features.union({feature})
            candidate_features_str = ', '.join(sorted(candidate_features))
            if candidate_features_str in df['features'].values:
                test_rmse = df.loc[df['features'] == candidate_features_str, 'Test RMSE'].values[0]
                train_rmse = df.loc[df['features'] == candidate_features_str, 'Training RMSE'].values[0]
                
                if test_rmse < best_rmse:
                    best_rmse = test_rmse
                    best_feature = feature
                    best_train_rmse = train_rmse  # 记录最佳训练RMSE
        
        if best_feature:
            selected_features.add(best_feature)
            remaining_features.remove(best_feature)
            current_rmse = best_rmse
            best_features = sorted(list(selected_features))
            print(f"Selected features: {best_features}, Training RMSE: {best_train_rmse}, Test RMSE: {best_rmse}")
        else:
            break
    
    return best_features

# 使用修改后的函数
selected_features_forward = forward_selection(df)

def backward_selection(df):
    """
    后向选择算法：从包含所有特征的模型开始，每一步删除一个特征，使得剩余特征的模型的测试RMSE最小。
    参数：
        df (DataFrame)：包含特征组合及其对应的训练和测试RMSE的数据框。
    返回：
        current_features (list)：选择的最佳特征组合列表。
    """
    features = set(['x1', 'x2', 'x3', 'x4'])
    current_features = features.copy()
    current_rmse = df.loc[df['features'] == ', '.join(sorted(features)), 'Test RMSE'].values[0]
    train_rmse = df.loc[df['features'] == ', '.join(sorted(features)), 'Training RMSE'].values[0]
    #sorted(features)：对 features 集合中的特征进行排序。这样可以确保特征的顺序一致，以便进行字符串匹配。
    #', '.join(sorted(features))：将排序后的特征列表连接成一个字符串，其中每个特征之间用逗号和空格分隔。例如，如果 features 包含 {'x1', 'x3'}，则生成的字符串为 'x1, x3'。
    #df['features'] == ', '.join(sorted(features))：在数据框 df 中找到 features 列与上述生成的字符串相匹配的行。
    #这将返回一个布尔系列（Boolean Series），其中匹配的行对应于 True，不匹配的行对应于 False。
    #df.loc[df['features'] == ', '.join(sorted(features))]：使用 .loc 索引器根据布尔系列从数据框 df 中选择匹配的行。
    #df.loc[df['features'] == ', '.join(sorted(features)), 'Training RMSE']：在选择的行中，提取 Training RMSE 列的值。
    #.values[0]：从提取的 Training RMSE 列中获取第一个值（因为我们期望只有一个匹配的行）。
    
    while len(current_features) > 1:
        best_rmse = current_rmse
        worst_feature = None
        
        for feature in current_features:
            candidate_features = current_features - {feature}
            candidate_features_str = ', '.join(sorted(candidate_features))
            if candidate_features_str in df['features'].values:
                test_rmse = df.loc[df['features'] == candidate_features_str, 'Test RMSE'].values[0]
                
                if test_rmse < best_rmse:
                    best_rmse = test_rmse
                    worst_feature = feature
                    best_train_rmse = train_rmse  # 记录最佳训练RMSE
        
        if worst_feature:
            current_features.remove(worst_feature)
            current_rmse = best_rmse
            print(f"Remaining features: {sorted(list(current_features))}, Training RMSE: {best_train_rmse}, Test RMSE: {best_rmse}")
        else:
            break
    
    return sorted(list(current_features))

# 使用修改后的函数
selected_features_backward = backward_selection(df)


print("Forward Selection:", selected_features_forward)
print("Backward Selection:", selected_features_backward)
