import numpy as np
from fitness import system_model
import time
class AntColony:
    def __init__(self, k, num_ants=100, iterations=100, alpha=1.0, beta=5.0, evaporation_rate=0.5, q=100):
        self.k = k
        self.num_ants = num_ants
        self.iterations = iterations
        self.alpha = alpha  # 信息素重要程度
        self.beta = beta  # 启发函数重要程度
        self.evaporation_rate = evaporation_rate  # 信息素挥发率
        self.q = q  # 信息素增加值

        # 初始化信息素矩阵
        self.pheromone_A = np.ones(31)  # 对应 A 的列选择
        self.pheromone_B = np.ones(31)  # 对应 B 的列选择

    def cal_Energy(self, A, B):
        fitness = system_model(A, B)
        return fitness

    def select_column(self, pheromone, heuristic, size):
        """根据信息素和启发度选取列"""
        probabilities = pheromone ** self.alpha * heuristic ** self.beta
        probabilities /= np.sum(probabilities)
        return np.random.choice(np.arange(size), p=probabilities)

    def heuristic_function(self):
        """启发函数，此处简单使用随机数，实际使用可替换为更复杂的启发式衡量"""
        return np.random.rand(31)

    def update_pheromone(self, ants):
        """更新信息素"""
        self.pheromone_A *= (1 - self.evaporation_rate)
        self.pheromone_B *= (1 - self.evaporation_rate)

        for ant in ants:
            reward = ant['value']
            for col in ant['idx_A']:
                self.pheromone_A[col] += self.q / reward  # 更新 A 的信息素

            for col in ant['idx_B']:
                self.pheromone_B[col] += self.q / reward  # 更新 B 的信息素

    def run(self):
        data = []
        best_value = -np.inf
        best_idx_A, best_idx_B = None, None

        for _ in range(self.iterations):
            ants = []
            heuristic_A = self.heuristic_function()
            heuristic_B = self.heuristic_function()
            for _ in range(self.num_ants):
                idx_A = [self.select_column(self.pheromone_A, heuristic_A, 31) for _ in range(self.k)]
                idx_B = [self.select_column(self.pheromone_B, heuristic_B, 31) for _ in range(self.k)]
                # 从矩阵 A 和 B 中选择对应的列
                # A_selected = self.A[:, idx_A]  # 选取的列
                # B_selected = self.B[:, idx_B]  # 选取的列

                # 计算适应度（根据选择的列计算 C 的值）
                product_value = self.cal_Energy(idx_A, idx_B)

                ants.append({'idx_A': idx_A, 'idx_B': idx_B, 'value': product_value})

                # 更新最佳解
                if product_value > best_value:
                    best_value = product_value
                    best_idx_A = idx_A
                    best_idx_B = idx_B
                    # 更新信息素
            self.update_pheromone(ants)
            data.append(best_value[0][0])
            print(idx_A)
            print(f"最优适应度: {best_value}")
        # 打开文件（如果文件不存在则会创建），以写入模式打开
        with open('ACO.txt', 'w') as file:
            for item in data:
                file.write(str(item) + '\n')
        return best_idx_A, best_idx_B, best_value

    # 示例矩阵 A 和 B
# 记录开始时间
start_time = time.time()
k = 8  # 要选择的列数

# 蚁群算法求解
ant_colony = AntColony(k)
idx_A_best, idx_B_best, max_value = ant_colony.run()

print("Optimal indices from A:", idx_A_best)
print("Optimal indices from B:", idx_B_best)
print("Maximum value of A[idx_A] @ B[idx_B]:", max_value)

# 记录结束时间
end_time = time.time()

# 计算运行时间
execution_time = end_time - start_time
print(f"程序运行时间: {execution_time:.6f} 秒")
