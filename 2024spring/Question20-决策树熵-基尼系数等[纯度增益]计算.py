import numpy as np
from typing import List, Tuple

def entropy(probabilities: np.ndarray) -> float:
    """计算熵"""
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

def gini_index(probabilities: np.ndarray) -> float:
    """计算基尼指数"""
    return 1 - np.sum(probabilities ** 2)

def classification_error(probabilities: np.ndarray) -> float:
    """计算分类误差率"""
    return 1 - np.max(probabilities)

def calculate_purity_gain(root_counts: np.ndarray, branch_counts: List[np.ndarray], method: str) -> float:
    """
    计算纯度增益
    
    参数:
    root_counts: 根节点的类别计数
    branch_counts: 分支节点的类别计数列表
    method: 纯度度量方法 ('entropy', 'gini', 或 'error')
    
    返回:
    纯度增益值
    """
    total_samples = np.sum(root_counts)
    root_probabilities = root_counts / total_samples

    if method == 'entropy':
        measure_func = entropy
    elif method == 'gini':
        measure_func = gini_index
    elif method == 'error':
        measure_func = classification_error
    else:
        raise ValueError("无效的方法。请选择 'entropy'、'gini' 或 'error'。")

    root_measure = measure_func(root_probabilities)

    weighted_branch_measure = sum(
        np.sum(branch) / total_samples * measure_func(branch / np.sum(branch))
        for branch in branch_counts
    )

    return root_measure - weighted_branch_measure

# 使用示例
if __name__ == "__main__":
    # 示例 1: 熵计算
    root_counts_1 = np.array([300, 100])
    branch_counts_1 = [np.array([175, 75]), np.array([125, 25])]
    method_1 = 'entropy'

    purity_gain_1 = calculate_purity_gain(root_counts_1, branch_counts_1, method_1)
    print(f"示例 1 - 熵增益: {purity_gain_1:.6f}")

    # 示例 2: 基尼指数计算
    root_counts_2 = np.array([300, 100])
    branch_counts_2 = [np.array([175, 75]), np.array([125, 25])]
    method_2 = 'gini'

    purity_gain_2 = calculate_purity_gain(root_counts_2, branch_counts_2, method_2)
    print(f"示例 2 - 基尼指数增益: {purity_gain_2:.6f}")

    # 示例 3: 分类误差率计算（使用不同的数据）
    root_counts_3 = np.array([200, 150, 50])
    branch_counts_3 = [np.array([80, 60, 10]), np.array([120, 90, 40])]
    method_3 = 'error'

    purity_gain_3 = calculate_purity_gain(root_counts_3, branch_counts_3, method_3)
    print(f"示例 3 - 分类误差率增益: {purity_gain_3:.6f}")

    # 您可以通过修改上面的参数来计算不同情况下的纯度增益