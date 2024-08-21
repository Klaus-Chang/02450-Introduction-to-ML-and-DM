import matplotlib.pyplot as plt
import numpy as np

# 创建数据点
x1 = np.linspace(0, 50, 100)
x2 = np.linspace(0, 25, 100)
X1, X2 = np.meshgrid(x1, x2)

# 决策边界函数
def decision_boundary(X1, X2):
    y = np.ones_like(X1)
    y[X1 < 20.95] = 1
    y[(X1 >= 20.95) & (X2 < 5.65)] = 1
    y[(X1 >= 20.95) & (X2 >= 5.65) & (X2 < 11.85)] = 0
    y[(X1 >= 20.95) & (X2 >= 11.85) & (X1 < 41.9)] = 0
    y[(X1 >= 20.95) & (X2 >= 11.85) & (X1 >= 41.9)] = 1
    return y

# 绘制决策边界
Y = decision_boundary(X1, X2)

plt.figure(figsize=(10, 6))
plt.contourf(X1, X2, Y, alpha=0.3, cmap='coolwarm')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Decision Boundaries from Decision Tree')
plt.axvline(x=20.95, color='blue', linestyle='--')
plt.axhline(y=5.65, color='red', linestyle='--')
plt.axhline(y=11.85, color='green', linestyle='--')
plt.axvline(x=41.9, color='purple', linestyle='--')
plt.show()
