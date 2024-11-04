import datetime
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间

import numpy as np
import random
import torch
from parl.utils import logger
from ddpg import DDPG
from common.utils import save_results, make_dir
from common.utils import plot_rewards, plot_axis
from env import Environment


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class Config:
    '''超参数'''

    def __init__(self):
        ################################## 环境超参数 ###################################
        self.algo_name = 'DDPG'  # 算法名称
        self.env_name = 'uav_v2'  # 环境名称
        self.device = torch.device("cpu")  # 检测GPU
        self.seed = 10  # 随机种子，置0则不设置随机种子
        self.train_eps = 10000  # 训练的回合数
        self.test_eps = 100  # 测试的回合数
        self.steps = 400  # 每回合的step

        ################################## 算法超参数 ###################################
        self.gamma = 0.9    # 折扣因子
        self.critic_lr = 1e-3  # 评论家网络的学习率
        self.actor_lr = 1e-3 # 演员网络的学习率
        self.memory_capacity = 2000000  # 经验回放的容量
        self.batch_size = 256  # mini-batch SGD中的批量大小
        self.target_update = 2  # 目标网络的更新频率
        self.hidden_dim = 256  # 网络隐藏层维度
        self.soft_tau = 1e-3  # 软更新参数
        self.NOISE = 0.05
        self.REWARD_SCALE = 0.1  # reward 缩放系数
        ################################# 保存结果相关参数 ################################
        self.result_path = "./outputs/" + self.env_name + \
                           '/' + curr_time + '/results/'  # 保存结果的路径
        self.model_path = "./outputs/" + self.env_name + \
                          '/' + curr_time + '/models/'  # 保存模型的路径
        self.save = True  # 是否保存图片

def env_agent_config(cfg,seed=0):
    env = Environment()
    #env = GridEnv()
    state_dim = env.state_dim
    action_dim = env.action_dim
    agent = DDPG(state_dim, action_dim, cfg)
    return env, agent

def train(cfg, env, agent):
    print('开始训练！')
    print(f'环境：{cfg.env_name}，算法：{cfg.algo_name}，设备：{cfg.device}')
    rewards = [] # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励

    for i_ep in range(cfg.train_eps):
        state, state_norm = env.reset()
        #print("start state=",state)
        ep_reward = 0
        total_reward = 0
        steps = 0
        uav_track = np.zeros([1000, 3])
        average_reward = []
        total_step = []
        for i_step in range(cfg.steps):
            i_step += 1
            steps += 1
            state_norm = np.array(state_norm).reshape(-1)
            batch_obs = np.expand_dims(state_norm, axis=0)
            action = agent.choose_action(batch_obs)
            action = (action + 1) / 2
            action = np.clip(np.random.normal(action, cfg.NOISE), 0, 1)
            next_state, next_state_norm, reward, done, x, y, Action1, Action2, st, energy = env.step(action[0], state)
            if i_step == cfg.steps - 1:
                reward = reward - 100
                done = True
            ep_reward += reward
            agent.memory.push(state_norm, action, cfg.REWARD_SCALE * reward, next_state_norm, done)
            uav_track[steps, :] = [x, y, 100]
            np.savetxt('./uav_track_hddpg/uav_track_{}'.format(i_ep), uav_track, fmt='%f')
            agent.update()
            state = next_state
            state_norm = next_state_norm
            total_reward += reward
            if done:
                # np.savetxt('./uav_track_hddpg/uav_track_{}'.format(i_ep), uav_track, fmt='%f')
                break
        average_reward.append(total_reward)
        total_step.append(steps)
        logger.info(
            'episode:{}   Train reward:{}  total step:{}'.format(
                i_ep, average_reward, total_step))
        rewards.append(ep_reward)

        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)

    print('完成训练！')
    return rewards, ma_rewards

def test(cfg, env, agent):
    print('开始测试！')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards = [] # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    s_norm = State_Normlization(env.area_height,env.area_width,env.z_max,env.z_min)

    for i_ep in range(cfg.test_eps):
        state = env.reset()
        ep_reward = 0

        for i_step in range(cfg.steps):
            i_step += 1
            action = agent.choose_action(state)
            next_state, reward, done, crash_over = env.step(action)
            if i_step == cfg.steps:
                print("{}步都没有出界呢！".format(cfg.steps))
                done = True
            ep_reward += reward
            state = next_state
            if crash_over:
                print("Episode finished after {} timesteps".format(i_step))
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.95*ma_rewards[-1]+0.05*ep_reward)
        else:
            ma_rewards.append(ep_reward)
        print(f"回合：{i_ep+1}/{cfg.test_eps}，奖励：{ep_reward:.1f}")
    print('完成测试！')
    return rewards, ma_rewards

if __name__ == "__main__":
    cfg = Config()
    # 训练
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)
    env, agent = env_agent_config(cfg, seed=1)
    rewards, ma_rewards = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)
    agent.save(path=cfg.model_path)
    save_results(rewards, ma_rewards, tag='train', path=cfg.result_path)
    plot_rewards(rewards, ma_rewards, cfg, tag="train")  # 画出结果

    # 测试
    # env,agent = env_agent_config(cfg, seed=0)
    # agent.load(path=cfg.model_path)
    # rewards,ma_rewards = test(cfg, env,agent)
    # save_results(rewards, ma_rewards, tag = 'test', path = cfg.result_path)
    # plot_rewards(rewards, ma_rewards, cfg, tag="test")  # 画出结果

