from itertools import combinations

# Define a function to calculate D and S for a given Q and Z
def calculate_D_S(Q, Z):
    D = 0
    S = 0
    for (i, j) in combinations(range(len(Q)), 2):
        if Q[i] == Q[j] and Z[i] == Z[j]:
            S += 1
        elif Q[i] != Q[j] and Z[i] != Z[j]:
            D += 1
    return D, S

# Given options
Q_A = [3, 1, 3, 1, 2, 1, 2, 1, 3, 1]
Z_A = [1, 3, 1, 2, 3, 3, 4, 2, 1, 1]

Q_B = [1, 1, 2, 1, 2, 1, 2, 1, 3, 1]
Z_B = [4, 1, 3, 2, 3, 3, 4, 2, 1, 1]

Q_C = [1, 1, 1, 1, 2, 1, 2, 1, 3, 1]
Z_C = [2, 3, 2, 2, 3, 3, 4, 2, 1, 1]

Q_D = [1, 2, 2, 1, 2, 1, 2, 1, 3, 1]
Z_D = [1, 3, 4, 2, 3, 3, 4, 2, 1, 1]

# Calculate D and S for each option
results = {
    "A": calculate_D_S(Q_A, Z_A),
    "B": calculate_D_S(Q_B, Z_B),
    "C": calculate_D_S(Q_C, Z_C),
    "D": calculate_D_S(Q_D, Z_D)
}


print(f"\nresults: {results}")
