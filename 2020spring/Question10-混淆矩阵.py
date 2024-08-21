#填入TP、TN、FP、FN的值
TP = 34
TN = 39
FP = 7
FN = 11

# 计算Precison和Recall
Precision = TP/(TP + FP)
Recall = TP/(TP + FN)

F1 = 2*Precision*Recall / (Precision + Recall)

# 打印F1的值
print(f"F1= {F1:.4f}")