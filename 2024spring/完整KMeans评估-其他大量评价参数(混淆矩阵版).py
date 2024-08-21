import numpy as np
from sklearn.metrics import adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, fowlkes_mallows_score, homogeneity_completeness_v_measure, jaccard_score
from sklearn.preprocessing import LabelEncoder

def calculate_metrics(confusion_matrix):
    # 从混淆矩阵重构标签
    true_labels = []
    pred_labels = []
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            true_labels.extend([i] * confusion_matrix[i, j])
            pred_labels.extend([j] * confusion_matrix[i, j])
    
    # 编码标签
    le = LabelEncoder()
    true_labels = le.fit_transform(true_labels)
    pred_labels = le.fit_transform(pred_labels)
    
    # 计算各种指标
    ari = adjusted_rand_score(true_labels, pred_labels)
    mi = mutual_info_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    fm_score = fowlkes_mallows_score(true_labels, pred_labels)
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(true_labels, pred_labels)
    jaccard = jaccard_score(true_labels, pred_labels, average='micro')
    
    return {
        "Adjusted Rand Index": ari,
        "Mutual Information": mi,
        "Normalized Mutual Information": nmi,
        "Fowlkes-Mallows Score": fm_score,
        "Homogeneity": homogeneity,
        "Completeness": completeness,
        "V-measure": v_measure,
        "Jaccard Index": jaccard
    }

# 从图中提取的混淆矩阵
confusion_matrix = np.array([
    [114, 0, 32],
    [0, 119, 0],
    [8, 0, 80]
])

metrics = calculate_metrics(confusion_matrix)

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")