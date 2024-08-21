import numpy as np
from sklearn.metrics import jaccard_score, adjusted_rand_score, mutual_info_score, normalized_mutual_info_score
from itertools import combinations

class ClusteringEvaluator:
    def __init__(self, true_labels, pred_labels):
        self.true_labels = np.array(true_labels)
        self.pred_labels = np.array(pred_labels)

    def jaccard_index_custom(self):
        n = len(self.true_labels)
        pairs = list(combinations(range(n), 2))
        
        S = sum(1 for i, j in pairs if self.true_labels[i] == self.true_labels[j] and 
                self.pred_labels[i] == self.pred_labels[j])
        D = sum(1 for i, j in pairs if self.true_labels[i] != self.true_labels[j] and 
                self.pred_labels[i] != self.pred_labels[j])
        
        return S / (n * (n - 1) / 2 - D)

    def jaccard_index_sklearn(self):
        return jaccard_score(self.true_labels, self.pred_labels, average='micro')

    def adjusted_rand_index(self):
        return adjusted_rand_score(self.true_labels, self.pred_labels)

    def mutual_information(self):
        return mutual_info_score(self.true_labels, self.pred_labels)

    def normalized_mutual_information(self):
        return normalized_mutual_info_score(self.true_labels, self.pred_labels)

    def evaluate_all(self):
        return {
            "Jaccard Index (Custom)": self.jaccard_index_custom(),
            "Jaccard Index (Sklearn)": self.jaccard_index_sklearn(),
            "Adjusted Rand Index": self.adjusted_rand_index(),
            "Mutual Information": self.mutual_information(),
            "Normalized Mutual Information": self.normalized_mutual_information()
        }

def parse_data(data_str):
    """解析字符串形式的数据为numpy数组"""
    return np.array([list(map(int, row.split())) for row in data_str.strip().split('\n')])

def find_closest_option(value, options):
    """找出最接近的选项"""
    return min(options, key=lambda x: abs(x - value))

# 使用示例
if __name__ == "__main__":
    # 数据输入（可以根据需要修改）
    data_str = """
    1 1 1 0 0
    1 1 1 0 0
    1 1 1 0 0
    1 1 1 0 0
    1 1 1 0 0
    0 1 1 0 0
    0 1 0 1 1
    1 1 1 0 0
    1 0 1 0 0
    0 0 1 1 1
    0 1 0 1 1
    """
    data = parse_data(data_str)

    true_labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    pred_labels = data[:, 1]  # 使用f2列作为预测标签

    evaluator = ClusteringEvaluator(true_labels, pred_labels)
    results = evaluator.evaluate_all()

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    # 对于特定问题的答案选择
    options = [0.0909, 0.5273, 0.7436, 0.7838]
    closest = find_closest_option(results["Jaccard Index (Custom)"], options)
    print(f"\nClosest option to Jaccard Index: {closest}")