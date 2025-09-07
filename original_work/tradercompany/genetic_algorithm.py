import numpy as np
import math
import random

crossoverrate = 0.9 #交差率
mutationrate = 0.01 #突然変異率
populations = 20    #個体群の大きさ
generations = 1     #世代
genoms = 8          #遺伝子の長さ

SEED = 2021
np.random.seed(SEED)

'''
個体の遺伝子 = [num_factors, delay_P, delay_Q, stock_P, stock_Q,
               activation_func, binary_operator, w]
各遺伝子の意味 = [使用する予測式の項数, どれだけ過去のデータを予測に用いるか,
                 予測に用いる銘柄番号, 活性化関数, 二項演算子, 各項の重み]
各遺伝子の型 = [int, int, int, int, int, activation_func[](numf_factors個),
               binary_operator[](num_factors個), int]
'''

#初期世代の作成------------------------------
class GeneticAlgorithmRecruiter: #遺伝子情報を持つ個体の定義

    def __init__(self, genom):
        self.genom = genom
        self.fitness = 0
        self.set_fitness()

    def set_fitness(self): #適応度の計算
        self.fitne4ss = self.genom.sum()

    def get_fitness(self):
        return self.fitness

    def mutate(self):
        '''遺伝子の突然変異'''
        tmp = self.genom.copy()
        i = np.random.randint(0, len(self.genom) - 1)
        tmp[i] = float(not self.genom[i])
        self.genom = tmp
        self.set_fitness()

def mutate(children):
    '''突然変異の関数'''
    for child in children:
        # 一定の確率で突然変異させる
        if np.random.rand() < mutationrate:
            child.mutate()
    return children


def create_generation(popurations, genoms):
    generation = []
    for i in range(populations):
        individual = GeneticAlgorithmRecruiter()
        generation.append(individual)
    return generation

#世代の各個体の適応度の評価------------------------------
def ga_solve(generation): #各世代の最高適応度と最低適応度をまとめ・表示
    best  = []
    worst = []

    for i in range(generations):
        best_ind = max(generation, key = GeneticAlgorithmRecruiter.get_fitness)
        best.append(best_ind.fitness)
        worst_ind = min(generation, key = GeneticAlgorithmRecruiter.get_fitness)
        worst.append(worst_ind.fitness)
        print("generation:" + str(i) \
                + ": Best fitness: " + str(best_ind.fitness) \
                + ". Worst fitness: " + str(worst_ind.fitness))

        print("Generation loop ended. The best individual: ")
        print(best_ind.genom)
        return best, worst

    # --- Step2. Selection (Roulette)
    selected = select_roulette(generation)

    # --- Step3. Crossover (two_point_copy)
    children = crossover(selected)

best, worst = ga_solve(generations)

#選択
def select_roulette(generation):
    '''選択の関数(ルーレット方式)'''
    selected = []
    weights = [ind.get_fitness() for ind in generation]
    norm_weights = [ind.get_fitness() / sum(weights) for ind in generation]
    selected = np.random.choice(generation, size=len(generation), p=norm_weights)
    return selected


def select_tournament(generation):
    '''選択の関数(トーナメント方式)'''
    selected = []
    for i in range(len(generation)):
        tournament = np.random.choice(generation, 3, replace=False)
        max_genom = max(tournament, key=GeneticAlgorithmRecruiter.get_fitness).genom.copy()
        selected.append(GeneticAlgorithmRecruiter(max_genom))
    return selected

#交叉
def multi_point_crossover(child1, child2):
    '''交叉の関数(二点交叉)
        input: 混ぜ合わせたい個体のペア
        output: 交叉後の個体のペア'''
    size = len(child1.genom)
    tmp1 = child1.genom.copy()
    tmp2 = child2.genom.copy()
    cxpoint1 = np.random.randint(1, size)
    cxpoint2 = np.random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    tmp1[cxpoint1:cxpoint2], tmp2[cxpoint1:cxpoint2] = tmp2[cxpoint1:cxpoint2].copy(), tmp1[cxpoint1:cxpoint2].copy()
    new_child1 = GeneticAlgorithmRecruiter(tmp1)
    new_child2 = GeneticAlgorithmRecruiter(tmp2)
    return new_child1, new_child2

def uniform_crossover(child1, child2):
    '''交叉の関数(一様交叉)
        input: 混ぜ合わせたい個体のペア
        output: 交叉後の個体のペア'''
    size = len(child1.genom)
    tmp1 = child1.genom.copy()
    tmp2 = child2.genom.copy()
    for i in range(0, size):
        if np.random.rand() < 0.5:
            continue
        else:
            tmp1[i], tmp2[i] = tmp2[i], tmp1[i]
    new_child1 = GeneticAlgorithmRecruiter(tmp1)
    new_child2 = GeneticAlgorithmRecruiter(tmp2)
    return new_child1, new_child2

def crossover(selected):
    '''交叉の関数'''
    children = []
    if populations % 2:
        selected.append(selected[0])
    for child1, child2 in zip(selected[::2], selected[1::2]):
        # 一定の確率で交叉を適用
        if np.random.rand() < crossoverrate:
            child1, child2 = multi_point_crossover(child1, child2)
        children.append(child1)
        children.append(child2)
    children = children[:populations]
    return children