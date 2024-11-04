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

class environment(object):

    def __init__(self):
        self.sum_energy = 0
        self.step0 = 1
        self.user_num = 8  # 用户数
        self.antenna_num = 32  # 天线数
        self.target2 = [160, 240]
        self.target = [200, 210]
        self.bds = []  # 建筑集合
        self.min_action = 0
        self.flag = 0
        self.state_dim = 284
        self.action_dim = 2
        # self.action_dim1 = [16, 16, 16, 16]
        # self.action_dim2 = [50, 50, 50, 50]
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.cable_long = 350
        self.k = 50
        self.P_max = 100  # 100w
        self.beta_0 = 1.42e-4
        self.alpha = 2
        self.sigma = 1e-17
        self.Z_uav = 50
        self.v_max = 10
        self.AS = np.array([0, 0])
        self.s = generate_complex_sinusoid(self.antenna_num) / np.sqrt(self.antenna_num)
        self.U = generate_complex_sinusoid_matrix(self.antenna_num) / np.sqrt(self.antenna_num)
        self.a = 1 / np.sqrt(self.antenna_num)
        self.n = generate_N(self.sigma, self.user_num)
        self.user_location = np.empty(shape=(self.user_num, 3))
        self.d = np.sqrt(np.sum((self.target - self.AS) ** 2))
        self.d2 = np.sqrt(np.sum((self.target2 - self.AS) ** 2))
        self.loc_uav = np.zeros(2)
        data = []
        with open('user_location') as f:
            for line in f.readlines():
                temp = line.split()
                data.append(temp)
        for i in range(self.user_num):
            self.user_location[i][0] = data[i][0]
            self.user_location[i][1] = data[i][1]
            self.user_location[i][2] = 0

        self.user_location = np.array(self.user_location).reshape(self.user_num, 3)
        # 计算功率分配矩阵
        n = np.arange(1, self.k + 1)
        p = self.P_max / self.k
        self.P = np.multiply(p, n)
    def system_model(self, uav_location, x):
        h = []
        total_energy = 0
        # h
        beta = np.zeros(self.user_num)
        for i in range(self.user_num):
            l_distance = np.sqrt(np.sum((self.user_location[i][0:2] - uav_location[0:2]) ** 2))
            distance = np.sqrt(l_distance ** 2 + self.Z_uav ** 2)
            beta[i] = self.beta_0 * (distance ** (-self.alpha))
            h.append(self.s * float(np.sqrt(self.antenna_num * beta[i])))
        h = np.array(h)
        h_k = h.reshape(self.user_num, self.antenna_num)
        for i in range(self.user_num):
            # p_k = self.P[Action1[i]]
            p_k = self.P[25 - (int(x[i]) + 25) % 50]
            # p_k = self.P[49]
            h = h_k[i].reshape(self.antenna_num, 1)
            h = h.T.conjugate()
            f = self.U[16 - (int(x[i+self.user_num]) + 16) % 32].reshape(self.antenna_num, 1)
            # f = self.U[31].reshape(self.antenna_num, 1)
            value = np.dot(h, f)
            # print(p_k * value * 10)
            # print(self.n[i] * 1e8)
            energy = abs(p_k * value * 10 + self.n[i] * 1e4) ** 2
            # if x[i+self.user_num] % 2 == 0:
            #     energy = energy * 1e6
            total_energy += energy

        return total_energy