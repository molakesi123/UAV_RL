import math
import gym
import random
import numpy as np
from sklearn import preprocessing

# 生成矩阵
def generate_complex_sinusoid_matrix(N):
    '''
    N (int): length of complex sinusoid in samples

    returns
    c_sin_matrix (numpy array): the generated complex sinusoid (length N)
    '''

    n = np.arange(N)
    n = np.expand_dims(n, axis=1)  # 扩充维度，将1D向量，转为2D矩阵，方便后面的矩阵相乘

    k = n

    m = n.T * k / N  # [N,1] * [1, N] = [N,N]

    S = np.exp(1j * np.pi * m)  # 计算矩阵 S

    return np.conjugate(S)


# 生成向量
def generate_complex_sinusoid(N):
    '''
    k (int): frequency index
    N (int): length of complex sinusoid in samples

    returns
    c_sin (numpy array): the generated complex sinusoid (length N)
    '''
    n = np.arange(N)
    c_sin = np.exp(1j * 2 * np.pi * 1 / 2 * n)

    return np.conjugate(c_sin)

# 生成复高斯随机变量-向量
def generate_N(sigma,user_num):
    mean = 0
    variance = sigma / 2
    Z = []
    X = np.random.normal(mean, np.sqrt(variance), user_num)
    Y = np.random.normal(mean, np.sqrt(variance), user_num)
    for i in range(user_num):
        Z.append(complex(X[i], Y[i]))
    n = np.array(Z)
    return n

user_num = 8  # 用户数
antenna_num = 32  # 天线数
k = 50
P_max = 100  # 100w
beta_0 = 1.42e-4
alpha = 2
sigma = 1e-17
Z_uav = 50
uav_location  = [50, 50, 50]
s = generate_complex_sinusoid(antenna_num) / np.sqrt(antenna_num)
U = generate_complex_sinusoid_matrix(antenna_num) / np.sqrt(antenna_num)
a = 1 / np.sqrt(antenna_num)
noise = generate_N(sigma, user_num)
user_location = np.empty(shape=(user_num, 3))
data = []
with open('user_location') as f:
    for line in f.readlines():
        temp = line.split()
        data.append(temp)
    for i in range(user_num):
        user_location[i][0] = data[i][0]
        user_location[i][1] = data[i][1]
        user_location[i][2] = 0

        user_location = np.array(user_location).reshape(user_num, 3)
        # 计算功率分配矩阵
        n = np.arange(1, k + 1)
        p = P_max / k
        P = np.multiply(p, n)


def system_model(x):
    h = []
    total_energy = 0
    # h
    beta = np.zeros(user_num)
    for i in range(user_num):
        l_distance = np.sqrt(np.sum((user_location[i][0:2] - uav_location[0:2]) ** 2))
        distance = np.sqrt(l_distance ** 2 + Z_uav ** 2)
        beta[i] = beta_0 * (distance ** (-alpha))
        h.append(s * float(np.sqrt(antenna_num * beta[i])))
    h = np.array(h)
    h_k = h.reshape(user_num, antenna_num)
    for i in range(user_num):
        # p_k = self.P[Action1[i]]
        p_k = P[abs(24 - (int(x[i]) + 26) % 50)]
        # p_k = self.P[int(x[i])]
        # p_k = self.P[25]
        h = h_k[i].reshape(antenna_num, 1)
        h = h.T.conjugate()
        # f = self.U[16 - (int(x[i+self.user_num]) + 16) % 32].reshape(self.antenna_num, 1)
        if abs(16 - (int(x[i+user_num]) + 16) % 32) % 2 == 0:
            f = U[abs(16 - (int(x[i+user_num]) + 16) % 32) + 1].reshape(antenna_num, 1)
        else:
            f = U[abs(16 - (int(x[i+user_num]) + 16) % 32)].reshape(antenna_num, 1)

        value = np.dot(h, f)

        energy = abs(p_k * value * 10 + noise[i] * 1e4) ** 2
        # if x[i+self.user_num] % 2 == 0:
        #     energy = energy * 1e6 / 3.5
        total_energy += energy
    return total_energy