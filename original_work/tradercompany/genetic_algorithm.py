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
            best_ind = max(generation, key = Individual.get_fitness)
            best.append(best_ind.fitness)
            worst_ind = min(generation, key = Individual.get_fitness)
            worst.append(worst_ind.fitness)
            print("generation:" + str(i) \
                    + ": Best fitness: " + str(best_ind.fitness) \
                    + ". Worst fitness: " + str(worst_ind.fitness))

            print("Generation loop ended. The best individual: ")
            print(best_ind.genom)
            return best, worst
