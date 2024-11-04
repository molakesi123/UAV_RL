# 粒子群算法求解31座城市TSP问题完整代码：
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from systemmodel import environment
from sklearn import preprocessing
# # 计算距离矩阵
# def clac_distance(X, Y):
#     """
#     计算两个城市之间的欧氏距离，二范数
#     :param X: 城市X的坐标.np.array数组
#     :param Y: 城市Y的坐标.np.array数组
#     :return:
#     """
#     distance_matrix = np.zeros((city_num, city_num))
#     for i in range(city_num):
#         for j in range(city_num):
#             if i == j:
#                 continue
#
#             distance = np.sqrt((X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2)
#             distance_matrix[i][j] = distance
#
#     return distance_matrix


# 定义总距离(路程即适应度值)
def fitness_func(env, x_i):
    """
    适应度函数
    :param distance_matrix: 城市距离矩阵
    :param x_i: PSO的一个解（路径序列）
    :return:
    """
    total_energy = env.system_model(x_i)

    return total_energy


# 定义速度更新函数
def get_v(w, V, p_best, g_best, x_i, c):
    """
    计算交换序列，即x2结果交换序列ss得到x1，对应PSO速度更新公式中的 r1(pbest-xi) 和 r2(gbest-xi)
    :param x_best: pbest or gbest
    :param x_i: 粒子当前的解
    :param r: 随机因子
    :return:
    """
    r1 = 0.7
    r2 = 0.8
    velocity_ss = []
    for i in range(len(x_i)):
        X_p = p_best[i] - x_i[i]
        v_p = X_p * c * r1
        X_g = g_best[i] - x_i[i]
        v_g = X_g * c * r2
        v = w * V[i] + v_p + v_g
        if v > 5:
            v = 5
        if v < -5:
            v = -5
        velocity_ss.append(v)

    return velocity_ss


# 定义位置更新函数
def get_location(x_i, v):
    """
    执行交换操作
    :param x_i:
    :param ss: 由交换子组成的交换序列
    :return:
    """
    for i in range(len(x_i)):
        x_i[i] = round(x_i[i] + v[i])
        if i < 8:
            if x_i[i] > 49:
                x_i[i] = 49
            if x_i[i] < 0:
                x_i[i] = 0
        else:
            if x_i[i] > 31:
                x_i[i] = 31
            if x_i[i] < 0:
                x_i[i] = 0
    return x_i

def init_pso():
    size = 100  # 粒子数量
    particle_v = np.zeros((size, 16), dtype=np.int64)
    pbest_init = np.zeros((size, 16), dtype=np.int64)
    for i in range(size):
        pbest_init[i][0:8] = np.random.choice(list(range(50)), size=8, replace=True)
        pbest_init[i][8:16] = np.random.choice(list(range(32)), size=8, replace=True)
        particle_v[i] = np.random.choice(list(range(4)), size=16, replace=True)
    # pbest_init[0] = [37, 43, 12, 8, 9, 11, 5, 15, 0, 16, 1, 12, 7, 13, 28, 6]
    # pbest_init[1] = [23, 41, 49, 30, 32, 22, 13, 41, 9, 7, 31, 29, 22, 25, 1, 0]
    # pbest_init[2] = [3, 4, 24, 49, 43, 12, 26, 16, 13, 19, 9, 18, 15, 0, 4, 25]
    # pbest_init[3] = [37, 19, 38, 8, 32, 34, 10, 23, 15, 15, 23, 25, 7, 19, 28, 10]
    # pbest_init[4] = [21, 6, 2, 12, 27, 21, 11, 7, 13, 8, 11, 12, 11, 20, 30, 4]
    # pbest_init[5] = [24, 10, 28, 20, 32, 12, 1, 30, 28, 9, 24, 18, 19, 1, 2, 12]
    # pbest_init[6] = [5, 17, 42, 20, 48, 22, 37, 13, 17, 18, 1, 21, 20, 10, 0, 23]
    # pbest_init[7] = [13, 15, 24, 9, 2, 7, 5, 36, 5, 24, 21, 8, 13, 27, 17, 17]
    # pbest_init[8] = [29, 11, 35, 29, 33, 2, 20, 19, 16, 22, 0, 29, 28, 23, 18, 31]
    # pbest_init[9] = [24, 49, 42, 20, 44, 15, 30, 27, 14, 19, 19, 27, 26, 11, 22, 7]
    # particle_v[0] = [1, 2, 0, 2, 1, 2, 0, 3, 0, 2, 0, 1, 2, 2, 0, 3]
    # particle_v[1] = [0, 1, 0, 0, 1, 3, 3, 2, 1, 0, 2, 3, 3, 2, 1, 1]
    # particle_v[2] = [3, 2, 3, 3, 2, 1, 0, 2, 1, 3, 3, 2, 3, 1, 0, 3]
    # particle_v[3] = [2, 2, 0, 0, 3, 3, 1, 1, 1, 3, 0, 0, 1, 1, 2, 0]
    # particle_v[4] = [3, 3, 1, 0, 1, 0, 0, 2, 0, 1, 2, 3, 0, 1, 3, 1]
    # particle_v[5] = [0, 3, 2, 2, 2, 0, 1, 2, 2, 2, 2, 3, 3, 0, 2, 3]
    # particle_v[6] = [1, 0, 1, 1, 3, 0, 1, 3, 1, 1, 2, 1, 0, 0, 3, 2]
    # particle_v[7] = [3, 1, 0, 2, 1, 3, 0, 0, 2, 1, 3, 0, 2, 2, 2, 0]
    # particle_v[8] = [0, 2, 2, 0, 1, 3, 0, 0, 0, 1, 2, 3, 1, 0, 0, 3]
    # particle_v[9] = [3, 1, 3, 2, 0, 3, 2, 0, 0, 0, 3, 3, 0, 3, 1, 2]
    return pbest_init, particle_v

def pso():
    env = environment()

    #参数设置
    w_max = 0.9  # 惯性权重
    w_min = 0.4
    sigma = 0.3  # 标准差
    c = 2
    size = 10  # 粒子数量
    iter_max_num = 1000  # 迭代次数
    fitness_value_lst = []

    # 计算每个粒子对应的适应度
    pbest, particle_v = init_pso()
    pbest_init = pbest
    pbest_fitness = np.zeros((size, 1))
    for i in range(size):
        pbest_fitness[i] = fitness_func(env, pbest_init[i])

    # 计算全局适应度和对应的gbest
    gbest = pbest_init[pbest_fitness.argmin()]
    gbest_fitness = pbest_fitness.max()

    # 记录算法迭代效果
    fitness_value_lst.append(gbest_fitness)

    # 迭代过程
    for i in range(iter_max_num):
        W = w_min + (w_max - w_min) * np.random.rand() + sigma * np.random.randn()
        # 控制迭代次数
        for j in range(size):
            # 遍历每个粒子
            pbest_i = pbest[j].copy()
            x_i = pbest_init[j].copy()

            # 计算交换序列，即 v = r1(pbest-xi) + r2(gbest-xi)
            V = get_v(W, particle_v[j], pbest_i, gbest, x_i, c)
            x_i = get_location(x_i, V)

            fitness_new = fitness_func(env, x_i)
            fitness_old = pbest_fitness[j]
            if fitness_new > fitness_old:
                pbest_fitness[j] = fitness_new
                pbest[j] = x_i

            gbest_fitness_new = pbest_fitness.max()
            gbest_new = pbest[pbest_fitness.argmax()]
            if gbest_fitness_new > gbest_fitness:
                gbest_fitness = gbest_fitness_new
                gbest = gbest_new
        fitness_value_lst.append(gbest_fitness)
    with open('fitness_PSO.txt', 'w') as file:
        for item in fitness_value_lst:
            file.write(str(item) + '\n')
    #输出迭代结果
    print("最优路线：", gbest)
    print("最优值：", gbest_fitness)

    # 绘图
    sns.set_style('whitegrid')
    plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示
    plt.rcParams['axes.unicode_minus'] = False
    #draw(gbest)
    plt.figure(2)
    plt.plot(fitness_value_lst)
    plt.title('优化过程')
    plt.ylabel('最优值')
    plt.xlabel('迭代次数({}->{})'.format(0, iter_max_num))
    plt.show()
    return gbest_fitness, gbest
# #
if __name__ == '__main__':
    data = pso()
