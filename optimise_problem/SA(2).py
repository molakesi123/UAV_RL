import numpy as np
from fitness import system_model
import time
# 设置随机种子以获得可重复的结果
np.random.seed(0)

# 初始化矩阵 C 和 D
C = np.random.rand(10, 10)  # 10x10 矩阵 C
D = np.random.rand(5, 5)  # 5x5 矩阵 D

# 定义矩阵 A 和 B 的形状
A_ROWS = 10  # 矩阵 A 的行数
A_COLS = 8  # 矩阵 A 的列数
B_ROWS = 5  # 矩阵 B 的行数
B_COLS = 8  # 矩阵 B 的列数（单列）


def calculate_fitness(A_indices, B_indices):
    """计算给定索引组合的适应度（矩阵乘积的最大值）"""
    fitness = system_model(A_indices, B_indices)
    return fitness  # 返回矩阵乘积的和


def simulated_annealing():
    # 初始设置
    current_A_indices = np.random.choice(range(31), size=A_COLS, replace=False)
    current_B_indices = np.random.choice(range(31), size=B_COLS, replace=False)

    current_fitness = calculate_fitness(current_A_indices, current_B_indices)

    # 模拟退火参数
    initial_temperature = 100.0
    final_temperature = 0.000001
    alpha = 0.999  # 温度衰减系数

    temperature = initial_temperature
    i = 0
    while temperature > final_temperature:
        # 随机生成新解
        i = i+1
        new_A_indices = current_A_indices.copy()
        new_B_indices = current_B_indices.copy()

        # 随机交换 A 中的一个列索引
        idx_A_to_change = np.random.randint(0, A_COLS)
        new_A_indices[idx_A_to_change] = np.random.choice(range(31), replace=False)

        # 随机交换 B 中的一个列索引
        idx_B_to_change = np.random.randint(0, B_COLS)
        new_B_indices[idx_B_to_change] = np.random.choice(range(31), replace=False)

        # 计算新解的适应度
        new_fitness = calculate_fitness(new_A_indices, new_B_indices)

        # 接受新解的概率
        acceptance_probability = np.exp((new_fitness - current_fitness) / temperature)

        # 按照概率决定是否跳转到新解
        if (new_fitness > current_fitness) or (np.random.rand() < acceptance_probability):
            current_A_indices = new_A_indices
            current_B_indices = new_B_indices
            current_fitness = new_fitness
            # 衰减温度
        temperature *= alpha
        print(i)
        print(f"最优适应度: {current_fitness}")
    return current_A_indices, current_B_indices, current_fitness


if __name__ == "__main__":
    # 记录开始时间
    start_time = time.time()
    best_A_indices, best_B_indices, best_fitness = simulated_annealing()
    print("Best A indices:", best_A_indices)
    print("Best B indices:", best_B_indices)
    print("Best fitness (maximum product):", best_fitness)
    # 记录结束时间
    end_time = time.time()

    # 计算运行时间
    execution_time = end_time - start_time
    print(f"程序运行时间: {execution_time:.6f} 秒")