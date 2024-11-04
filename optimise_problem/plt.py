import matplotlib.pyplot as plt
import numpy as np
# 文件列表和对应的算法名称
files = ['GA.txt', 'ACO.txt', 'PSO.txt']
labels = ['GA', 'ACO', 'PSO']
colors = ['r', 'g', 'b']  # 红、绿、蓝、紫

# 创建图形
plt.figure(figsize=(10, 6))

# 读取每个文件并绘制曲线
for file, label, color in zip(files, labels, colors):
    iterations = []
    fitness_values = []

    # 读取数据
    with open(file, 'r') as f:
        for line in f:
            data = line.split()
            fitness_values.append(float(data[0]))

            # 绘制曲线
    plt.plot(fitness_values, label=label, color=color)

# 添加图例、标题和坐标轴标签
plt.legend()
plt.title('Algorithm Performance Comparison')
plt.xlabel('Iterations')
plt.ylabel('Fitness Value')
plt.grid()
plt.tight_layout()

# 显示图形
plt.show()