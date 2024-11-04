'''
main(continuous)
PPO
antenna_num = 16
user = [4,4]
self.state_dim = 156
self.action_dim = 10
discrete_action = True
'''
import math
import gym
from gym.utils import seeding
import random
import numpy as np
from sklearn import preprocessing

class building():
    def __init__(self, x, y, l, w, h):
        self.x = x  # 建筑中心x坐标
        self.y = y  # 建筑中心y坐标
        self.l = l  # 建筑长半值
        self.w = w  # 建筑宽半值
        self.h = h  # 建筑高度
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

def generate_N(sigma,user_num):
    # 生成复高斯随机变量-向量
    mean = 0
    variance = sigma / 2
    Z = []
    X = np.random.normal(mean, np.sqrt(variance), user_num)
    Y = np.random.normal(mean, np.sqrt(variance), user_num)
    for i in range(user_num):
        Z.append(complex(X[i], Y[i]))
    n = np.array(Z)
    return n

class Environment(object):

    def __init__(self):
        self.sum_energy = 0
        self.step0 = 1
        self.user_num = 8  # 用户数
        self.antenna_num = 16  # 天线数
        self.target2 = [160, 240]
        self.target = [200, 210]
        self.bds = []  # 建筑集合
        self.min_action = 0
        self.flag = 0
        self.state_dim = 156
        self.action_dim = 10
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
        self.uav_location = np.empty(2)
        self.n = generate_N(self.sigma, self.user_num)
        self.user_location = np.empty(shape=(self.user_num, 3))
        self.d = np.sqrt(np.sum((self.target - self.AS) ** 2))
        self.d2 = np.sqrt(np.sum((self.target2 - self.AS) ** 2))
        self.loc_uav = np.zeros(2)
        self.map = np.zeros((500, 500, 130))
        self.map1 = np.zeros((500, 500, 130))
        self.map2 = np.zeros((500, 500, 130))
        self.map3 = np.zeros((500, 500, 130))
        self.map4 = np.zeros((500, 500, 130))
        self.map5 = np.zeros((500, 500, 130))
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

        self.seed()

    def seed(self, seed=None):  # seed设置为任意整数后，随机值固定，如果设置随机值固定
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # 定义每一个episode的初始状态，实现随机初始化
    def reset(self):
        self.sum_energy = 0
        self.step0 = 1
        self.loc_uav = np.zeros(2)
        #距离
        self.d2 = np.sqrt(np.sum((self.target2 - self.AS) ** 2))

        self.flag = 0
        # 生成建筑物
        self.bds.append(
            building(40, 80, 30, 25, 110))
        self.bds.append(
            building(117, 134, 50, 20, 110))
        self.bds.append(
            building(26, 174, 20, 20, 110))
        self.bds.append(
            building(150, 35, 20, 30, 110))
        self.bds.append(
            building(230, 101, 20, 80, 120))
        self.bds.append(
            building(115, 227, 20, 20, 120))
        self.bds.append(
            building(210, 200, 40, 15, 120))
        for i in range(7):
            self.map[self.bds[i].x - self.bds[i].l:self.bds[i].x + self.bds[i].l,
            self.bds[i].y - self.bds[i].w:self.bds[i].y + self.bds[i].w, 0:self.bds[i].h] = 1
        self.map1[150:175, 70:114, 0:100] = 1
        self.map2[50:100, 25:50, 0:100] = 1
        self.map3[100:125, 75:100, 0:100] = 1
        self.map4[175:200, 100:125, 0:100] = 1
        self.map5[175:200, 170:200, 0:100] = 1
        obs = []

        # 初始化无人机的位置
        uav_location = np.array([0, 0, 50])
        for i in range(3):
            obs.append(uav_location[i])

        # 基站位置
        for i in range(self.user_num):
            obs.append(self.user_location[i][0])
            obs.append(self.user_location[i][1])
            obs.append(self.user_location[i][2])
        # h
        beta = np.zeros(self.user_num)
        for i in range(self.user_num):
            l_distance = np.sqrt(np.sum((self.user_location[i][0:2] - uav_location[0:2]) ** 2))
            distance = np.sqrt(l_distance ** 2 + self.Z_uav ** 2)
            beta[i] = self.beta_0 * (distance ** (-self.alpha))
            h = self.s * float(np.sqrt(self.antenna_num * beta[i]))
            for k in range(self.antenna_num):
                obs.append(h[k])
        obs.append(self.d2)
        obs = np.array(obs)
        # 复数归一化
        max_imag1 = obs[42].imag * obs[42].imag
        min_imag1 = obs[27].imag * obs[27].imag
        d1 = max_imag1 - min_imag1
        max_imag2 = obs[58].imag * obs[58].imag
        min_imag2 = obs[43].imag * obs[43].imag
        d2 = max_imag2 - min_imag2
        max_imag3 = obs[74].imag * obs[74].imag
        min_imag3 = obs[59].imag * obs[59].imag
        d3 = max_imag3 - min_imag3
        max_imag4 = obs[90].imag * obs[90].imag
        min_imag4 = obs[75].imag * obs[75].imag
        d4 = max_imag4 - min_imag4
        max_imag5 = obs[106].imag * obs[106].imag
        min_imag5 = obs[91].imag * obs[91].imag
        d5 = max_imag5 - min_imag5
        max_imag6 = obs[122].imag * obs[122].imag
        min_imag6 = obs[107].imag * obs[107].imag
        d6 = max_imag6 - min_imag6
        max_imag7 = obs[138].imag * obs[138].imag
        min_imag7 = obs[123].imag * obs[123].imag
        d7 = max_imag7 - min_imag7
        max_imag8 = obs[154].imag * obs[154].imag
        min_imag8 = obs[139].imag * obs[139].imag
        d8 = max_imag8 - min_imag8
        obs_normalization = np.empty(shape=(self.state_dim,))
        obs_normalization[0:3] = obs[0:3] / 250
        obs_normalization[3:27] = obs[3:27] / 250
        obs_normalization[155] = obs[155] / self.d2
        for i in range(16):
            imag = obs[i + 27].imag * obs[i + 27].imag
            obs_normalization[i + 27] = (imag - min_imag1) / d1
            imag = obs[i + 43].imag * obs[i + 43].imag
            obs_normalization[i + 43] = (imag - min_imag2) / d2
            imag = obs[i + 59].imag * obs[i + 59].imag
            obs_normalization[i + 59] = (imag - min_imag3) / d3
            imag = obs[i + 75].imag * obs[i + 75].imag
            obs_normalization[i + 75] = (imag - min_imag4) / d4
            imag = obs[i + 91].imag * obs[i + 91].imag
            obs_normalization[i + 91] = (imag - min_imag4) / d5
            imag = obs[i + 107].imag * obs[i + 107].imag
            obs_normalization[i + 107] = (imag - min_imag4) / d6
            imag = obs[i + 123].imag * obs[i + 123].imag
            obs_normalization[i + 123] = (imag - min_imag4) / d7
            imag = obs[i + 139].imag * obs[i + 139].imag
            obs_normalization[i + 139] = (imag - min_imag4) / d8
        # obs_normalization[15:79] = obs[15:79]
        # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))  # 归一化函数
        # 标准化用户坐标
        # user = min_max_scaler.fit_transform(a)
        return obs, obs_normalization

    def step(self, action, obs):
        reward = 0
        uav_user = []
        Action1 = []
        Action2 = []
        # action = (action + 1) / 2
        # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        x, y = obs[0], obs[1]
        self.loc_uav[0] = x
        self.loc_uav[1] = y
        # x = 140
        # y = 188
        # print('x:', x)
        # print('y:', y)
        next_obs = []
        # 输入到无人机模型中的action为[0, pi/2]
        v = self.v_max * action[0]
        # v = 0
        # v = self.v_max
        theta = action[1] * math.pi
        for i in range(2, 6):
            Action1.append(round(action[i] * 24.49) + 25)
        Action1 = np.array(Action1)
        for i in range(6, 10):
            Action2.append(round(action[i] * 15.49))
        Action2 = np.array(Action2)
        self.uav_location[0] = x + v * math.cos(theta)
        self.uav_location[1] = y + v * math.sin(theta)
        if self.flag == 2:
            d_orgin = np.sqrt(np.sum((self.target - self.loc_uav) ** 2))
            d_next = np.sqrt(np.sum((self.target - self.uav_location) ** 2))
            Ddis = d_orgin - d_next
            r = (self.d / d_next) * Ddis / 120 + 1
        else:
            d_orgin = np.sqrt(np.sum((self.target2 - self.loc_uav) ** 2))
            d_next = np.sqrt(np.sum((self.target2 - self.uav_location) ** 2))
            Ddis = d_orgin - d_next
            r = (self.d2 / d_next) * Ddis / 71
        next_obs.append(self.uav_location[0])
        next_obs.append(self.uav_location[1])
        next_obs.append(self.Z_uav)
        next_obs[3:27] = obs[3:27]
        beta = np.zeros(self.user_num)
        # if self.map1[int(x + v * math.cos(theta)), int(y + v * math.sin(theta)), 50] == 1:
        #     reward = 10
        #     self.map1[150:175, 70:114, 0:100] = 0
        # if self.map2[int(x + v * math.cos(theta)), int(y + v * math.sin(theta)), 50] == 1:
        #     reward = 10
        #     self.map2[50:100, 25:50, 0:100] = 0
        # if self.map3[int(x + v * math.cos(theta)), int(y + v * math.sin(theta)), 50] == 1:
        #     reward = 10
        #     self.map3[100:125, 75:100, 0:100] = 0
        # if self.map4[int(x + v * math.cos(theta)), int(y + v * math.sin(theta)), 50] == 1:
        #     reward = 10
        #     self.map4[175:200, 100:125, 0:100] = 0
        # if self.map5[int(x + v * math.cos(theta)), int(y + v * math.sin(theta)), 50] == 1:
        #     reward = 10
        #     self.map5[175:200, 100:125, 0:100] = 0
        for i in range(self.user_num):
            l_distance = np.sqrt(np.sum((self.user_location[i][0:2] - self.uav_location[0:2]) ** 2))
            distance = np.sqrt(l_distance ** 2 + self.Z_uav ** 2)
            uav_user.append(distance)
            beta[i] = self.beta_0 * (distance ** (-self.alpha))
            h = self.s * float(np.sqrt(self.antenna_num * beta[i]))
            for k in range(self.antenna_num):
                next_obs.append(h[k])
        next_obs.append(d_next)
        next_obs = np.array(next_obs)
        # uav_AS_l = np.sqrt(np.sum((self.uav_location - self.AS) ** 2))
        # uav_AS = np.sqrt(self.Z_uav ** 2 + uav_AS_l ** 2)
        # if uav_AS > self.cable_long:
        #     reward = -100
        #     self.safe = 1
        h_k = next_obs[27:155].reshape(8, 16)
        if x + v * math.cos(theta) > 250 or x + v * math.cos(theta) < 0:
            reward = -100
        elif y + v * math.sin(theta) > 250:
            reward = -100
        elif self.map[int(x + v * math.cos(theta)), int(y + v * math.sin(theta)), 50] == 1:
            reward = -100
        else:
            if self.flag == 0:
                self.step0 += 1
                count = 0
                for i in range(4):
                    p_k = self.P[Action1[i]]
                    # p_k = self.P[27]
                    h = h_k[i].reshape(self.antenna_num, 1)
                    h = h.T.conjugate()
                    f = self.U[Action2[i]].reshape(16, 1)
                    # f = self.U[14].reshape(16, 1)
                    value = np.dot(h, f)
                    # print(p_k * value * 10)
                    # print(self.n[i] * 1e8)
                    energy = abs(p_k * value * 10 + self.n[i] * 1e4) ** 2
                    if Action2[i] % 2 == 0:
                        energy = energy * 1e6
                    reward = reward + energy
                    self.sum_energy += energy
                    # reward = reward + (1 - (1 - energy / 0.37) ** 0.4)
                    if uav_user[i] < 65:
                        count += 1
                if count == 4:
                    reward += 100
                    self.flag = 1
            if self.flag == 1:
                count = 0
                for i in range(4):
                    p_k = self.P[Action1[i]]
                    # p_k = self.P[49]
                    h = h_k[i+4].reshape(self.antenna_num, 1)
                    h = h.T.conjugate()
                    f = self.U[Action2[i]].reshape(16, 1)
                    # f = self.U[15].reshape(16, 1)
                    value = np.dot(h, f)
                    # print(p_k * value * 10)
                    # print(self.n[i] * 1e8)
                    energy = abs(p_k * value * 10 + self.n[i+4] * 1e4) ** 2
                    if Action2[i] % 2 == 0:
                        energy = energy * 1e6
                    reward = reward + energy + 0.001
                    # print(ep)
                    # print('x:',self.uav_location[0])
                    # print('y:',self.uav_location[1])
                    # print(energy)
                    # reward = reward + (1 - (1 - energy / 0.37) ** 0.4)
                    if uav_user[i + 4] < 65:
                        count += 1
                if count == 4:
                    reward += 100
                    self.flag = 2
                # reward += 1
        # reward = reward + (1 - (1 - Ddistance) ** 0.4)
        # print(self.flag)
        # print(reward)
        reward = (reward / 1.44) + r
        reward = float(reward)
        if x + v * math.cos(theta) > 200 and y + v * math.sin(theta) > 200:
            reward += 1000
        # 无人机的新坐标
        x = x + v * math.cos(theta)
        y = y + v * math.sin(theta)
        # 复数归一化
        max_imag1 = next_obs[42].imag * next_obs[42].imag
        min_imag1 = next_obs[27].imag * next_obs[27].imag
        d1 = max_imag1 - min_imag1
        max_imag2 = next_obs[58].imag * next_obs[58].imag
        min_imag2 = next_obs[43].imag * next_obs[43].imag
        d2 = max_imag2 - min_imag2
        max_imag3 = next_obs[74].imag * next_obs[74].imag
        min_imag3 = next_obs[59].imag * next_obs[59].imag
        d3 = max_imag3 - min_imag3
        max_imag4 = next_obs[90].imag * next_obs[90].imag
        min_imag4 = next_obs[75].imag * next_obs[75].imag
        d4 = max_imag4 - min_imag4
        max_imag5 = next_obs[106].imag * next_obs[106].imag
        min_imag5 = next_obs[91].imag * next_obs[91].imag
        d5 = max_imag5 - min_imag5
        max_imag6 = next_obs[122].imag * next_obs[122].imag
        min_imag6 = next_obs[107].imag * next_obs[107].imag
        d6 = max_imag6 - min_imag6
        max_imag7 = next_obs[138].imag * next_obs[138].imag
        min_imag7 = next_obs[123].imag * next_obs[123].imag
        d7 = max_imag7 - min_imag7
        max_imag8 = next_obs[154].imag * next_obs[154].imag
        min_imag8 = next_obs[139].imag * next_obs[139].imag
        d8 = max_imag8 - min_imag8
        next_obs_normalization = np.empty(shape=(self.state_dim,))
        next_obs_normalization[0:3] = next_obs[0:3] / 250
        next_obs_normalization[3:27] = next_obs[3:27] / 250
        next_obs_normalization[155] = next_obs[155] / self.d2
        for i in range(16):
            imag = next_obs[i + 27].imag * next_obs[i + 27].imag
            next_obs_normalization[i + 27] = (imag - min_imag1) / d1
            imag = next_obs[i + 43].imag * next_obs[i + 43].imag
            next_obs_normalization[i + 43] = (imag - min_imag2) / d2
            imag = next_obs[i + 59].imag * next_obs[i + 59].imag
            next_obs_normalization[i + 59] = (imag - min_imag3) / d3
            imag = next_obs[i + 75].imag * next_obs[i + 75].imag
            next_obs_normalization[i + 75] = (imag - min_imag4) / d4
            imag = next_obs[i + 91].imag * next_obs[i + 91].imag
            next_obs_normalization[i + 91] = (imag - min_imag4) / d5
            imag = next_obs[i + 107].imag * next_obs[i + 107].imag
            next_obs_normalization[i + 107] = (imag - min_imag4) / d6
            imag = next_obs[i + 123].imag * next_obs[i + 123].imag
            next_obs_normalization[i + 123] = (imag - min_imag4) / d7
            imag = next_obs[i + 139].imag * next_obs[i + 139].imag
            next_obs_normalization[i + 139] = (imag - min_imag4) / d8
        # done = (uav_AS > self.cable_long) or (self.map[int(x), int(y), 100] == 1)
        done = (x < 0 or x > 250) or (y > 250) or (self.map[int(x), int(y), 100] == 1) or (x > 200 and y > 200)
        return next_obs, next_obs_normalization, reward, done, x, y, Action1, Action2, self.step0, self.sum_energy

