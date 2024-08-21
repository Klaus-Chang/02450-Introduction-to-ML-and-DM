#填入TP、TN、FP、FN的值
P_from_Africa = 0.154

#WATCH THIS
P_notfrom_Africa = 1 -0.154

P_over1000_fromA = 0.286
P_over1000_notfromA = 0.688

P_over1000_sum = P_over1000_fromA * P_from_Africa + P_over1000_notfromA * P_notfrom_Africa

# 计算 P(A|B)
P_A_given_B = (P_over1000_fromA * P_from_Africa) / P_over1000_sum
print(P_A_given_B)

print(P_A_given_B)