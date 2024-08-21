from itertools import combinations
from sklearn.metrics import jaccard_score, adjusted_rand_score

# Define a function to calculate Jaccard Index and Adjusted Rand Index
def calculate_jaccard_ari(Q, Z):
    # Jaccard Index
    jaccard = jaccard_score(Q, Z, average='macro')

    # Adjusted Rand Index
    ari = adjusted_rand_score(Q, Z)

    return jaccard, ari

# Given options
Q_A = [3, 1, 3, 1, 2, 1, 2, 1, 3, 1]
Z_A = [1, 3, 1, 2, 3, 3, 4, 2, 1, 1]

Q_B = [1, 1, 2, 1, 2, 1, 2, 1, 3, 1]
Z_B = [4, 1, 3, 2, 3, 3, 4, 2, 1, 1]

Q_C = [1, 1, 1, 1, 2, 1, 2, 1, 3, 1]
Z_C = [2, 3, 2, 2, 3, 3, 4, 2, 1, 1]

Q_D = [1, 2, 2, 1, 2, 1, 2, 1, 3, 1]
Z_D = [1, 3, 4, 2, 3, 3, 4, 2, 1, 1]

# Calculate Jaccard Index and ARI for each option
results_jaccard_ari = {
    "A": calculate_jaccard_ari(Q_A, Z_A),
    "B": calculate_jaccard_ari(Q_B, Z_B),
    "C": calculate_jaccard_ari(Q_C, Z_C),
    "D": calculate_jaccard_ari(Q_D, Z_D)
}

print(f"\nresults_jaccard_ari: {results_jaccard_ari}")