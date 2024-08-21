import numpy as np

def calculate_distances_and_cosine_similarity(v1, v2):
    # 计算2-范数距离
    norm_distance = np.linalg.norm(v1 - v2)
    
    # 计算余弦相似度
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cosine_similarity = dot_product / (norm_v1 * norm_v2)
    
    return norm_distance, cosine_similarity

# 定义向量 x35 和 x53
x35 = np.array([-1.24, -0.26, -1.04])
x53 = np.array([-0.60, -0.86, -0.50])

# 计算距离和相似度
distance, cosine_sim = calculate_distances_and_cosine_similarity(x35, x53)

print("2-范数距离：", distance)
print("余弦相似度：", cosine_sim)
