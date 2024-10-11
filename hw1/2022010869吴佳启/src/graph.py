import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 从结果文件中读取数据
data = np.loadtxt('results.txt')

# 提取 x、y 和结果值
x = data[:, 0]
y = data[:, 1]
results = data[:, 2]

# 获取 x_values 和 y_values 的范围
x_values = np.unique(x)
y_values = np.unique(y)

# 重塑结果数组以匹配 x 和 y 的形状
results = results.reshape((len(x_values), len(y_values)))

# 创建 3D 图形对象
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 生成 x 和 y 的网格
X, Y = np.meshgrid(x_values, y_values)

# 绘制三维热力图
surf = ax.plot_surface(X, Y, results, cmap='hot')
fig.colorbar(surf, label='Line accuracy')
ax.set_xlabel('LAMBDA')
ax.set_ylabel('ALPHA')
ax.set_zlabel('Line accuracy')
ax.set_title('trigrams')

plt.show()
