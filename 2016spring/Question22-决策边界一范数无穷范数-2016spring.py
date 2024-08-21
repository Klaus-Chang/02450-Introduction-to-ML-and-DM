import numpy as np
import matplotlib.pyplot as plt

def plot_l1_norm(c):
    x = np.linspace(-c, c, 400)
    y1 = c - np.abs(x)
    y2 = -c + np.abs(x)
    plt.plot(x, y1, 'orange')
    plt.plot(x, y2, 'orange')
    plt.fill_between(x, y1, y2, where=(y1 > y2), color='orange', alpha=0.3)

def plot_linf_norm(c):
    plt.plot([-c, -c], [-c, c], 'green')
    plt.plot([c, c], [-c, c], 'green')
    plt.plot([-c, c], [c, c], 'green')
    plt.plot([-c, c], [-c, -c], 'green')
    plt.fill_between([-c, c], [-c, -c], [c, c], color='green', alpha=0.3)

plt.figure(figsize=(6, 6))
plot_l1_norm(0.5)
plot_linf_norm(3/8)
plt.xlim(-0.6, 0.6)
plt.ylim(-0.6, 0.6)
plt.gca().set_aspect('equal', adjustable='box')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.show()
