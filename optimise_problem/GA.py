import random
from fitness import system_model
import time
# 定义个体类，代表种群中的一个个体
class Individual:
    def __init__(self, genes):
        self.genes = genes  # 个体的基因序列
        self.fitness = self.calculate_fitness()  # 个体的适应度

    def calculate_fitness(self):
        # 计算适应度函数，这里以基因的平方和为例
        # 适应度函数应根据具体问题进行定义
        fitness = system_model(self.genes[0:8], self.genes[8:16])
        return fitness

# 初始化种群
def initialize_population(size, gene_length):
    # size: 种群的大小
    # gene_length: 个体基因序列的长度
    # 生成初始种群，每个个体由随机生成的基因序列组成
    return [Individual([random.randint(0, 31) for _ in range(gene_length)]) for _ in range(size)]

# 选择过程
def selection(population, num_parents):
    # 根据适应度排序，选择适应度最高的个体作为父母
    # population: 当前种群
    # num_parents: 选择的父母数量
    sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)
    return sorted_population[:num_parents]

# 交叉过程
def crossover(parent1, parent2):
    # 单点交叉
    # parent1, parent2: 选择的两个父本个体
    # 随机选择交叉点，交换父本基因，生成两个子代
    point = random.randint(1, len(parent1.genes) - 1)
    child1_genes = parent1.genes[:point] + parent2.genes[point:]
    child2_genes = parent2.genes[:point] + parent1.genes[point:]
    return Individual(child1_genes), Individual(child2_genes)

# 变异过程
def mutation(individual, mutation_rate=0.01):
    # 对个体的基因序列进行随机变异
    # individual: 要变异的个体
    # mutation_rate: 变异概率
    for i in range(len(individual.genes)):
        if random.random() < mutation_rate:
            # 对每个基因位以一定的概率进行增减操作
            individual.genes[i] += random.randint(-1, 1)
    # 更新个体的适应度
    individual.fitness = individual.calculate_fitness()

# 遗传算法主函数
def genetic_algorithm(population_size, gene_length, num_generations):
    # population_size: 种群大小
    # gene_length: 基因长度
    # num_generations: 进化代数
    data = []
    # 初始化种群
    population = initialize_population(population_size, gene_length)
    for _ in range(num_generations):
        # 选择
        parents = selection(population, population_size // 2)
        next_generation = []
        # 生成新一代
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = crossover(parent1, parent2)
            mutation(child1)
            mutation(child2)
            next_generation.extend([child1, child2])
        population = next_generation
        # 每一代选出适应度最高的个体
        best_individual = max(population, key=lambda x: x.fitness)
        data.append(best_individual.fitness[0][0])
        print(f"最优适应度: {best_individual.fitness}")
    # 打开文件（如果文件不存在则会创建），以写入模式打开
    with open('GA.txt', 'w') as file:
        for item in data:
            file.write(str(item) + '\n')
    return best_individual

# 运行算法
# 记录开始时间
start_time = time.time()
best = genetic_algorithm(100, 16, 100)
print(f"最优个体基因: {best.genes}")
# 记录结束时间
end_time = time.time()

# 计算运行时间
execution_time = end_time - start_time
print(f"程序运行时间: {execution_time:.6f} 秒")

