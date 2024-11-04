import numpy as np
from fitness2 import system_model
import time
class Particle:
    def __init__(self, dimension):
        # 粒子的当前热度
        self.position = np.random.choice(range(31), size=dimension, replace=False)
        self.velocity = np.random.choice(range(4), size=dimension, replace=True)
        self.best_position = self.position.copy()
        self.best_value = -np.inf

    def evaluate(self):
        value = system_model(self.position)
        return value


def pso(num_particles=100, max_iter=100):
    data = []
    dimension = 16
    # 初始化粒子群
    particles = [Particle(dimension) for _ in range(num_particles)]
    global_best_position = None
    global_best_value = -np.inf

    for iteration in range(max_iter):
        for particle in particles:
            current_value = particle.evaluate()

            # 更新粒子的历史最佳
            if current_value > particle.best_value:
                particle.best_value = current_value
                particle.best_position = particle.position.copy()

                # 更新全局最佳
            if current_value > global_best_value:
                global_best_value = current_value
                global_best_position = particle.position.copy()

                # 更新粒子的位置和速度
        for particle in particles:
            inertia_weight = 0.5
            cognitive_component = 1.5 * np.random.rand() * (particle.best_position - particle.position)
            social_component = 1.5 * np.random.rand() * (global_best_position - particle.position)
            particle.velocity = inertia_weight * particle.velocity + cognitive_component + social_component

            # 更新粒子的运动控制
            particle.position = particle.position + particle.velocity
            particle.position = np.clip(particle.position, 0, 31).astype(int)  # 确保在有效范围内
        # 要写入的内容
        data.append(global_best_value[0][0])

        print(f"最优适应度: {global_best_value}")
    # 返回找到的最佳的矩阵乘积
    # 打开文件（如果文件不存在则会创建），以写入模式打开
    with open('PSO.txt', 'w') as file:
        for item in data:
            file.write(str(item) + '\n')
    return global_best_position, global_best_value

# 记录开始时间
start_time = time.time()

position, max_value = pso()
print("Optimal position A:\n", position)
print("Maximum value of A @ B:", max_value)
# 记录结束时间
end_time = time.time()

# 计算运行时间
execution_time = end_time - start_time
print(f"程序运行时间: {execution_time:.6f} 秒")
