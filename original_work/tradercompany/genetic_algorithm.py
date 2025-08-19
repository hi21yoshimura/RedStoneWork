import numpy as np
import math
import random

crossoverrate = 0.9 #交差率
mutationrate = 0.01 #突然変異率
populations = 20     #個体群の大きさ
generations = 1      #世代

'''
個体の遺伝子 = [num_factors, delay_P, delay_Q, stock_P, stock_Q,
               activation_func, binary_operator, w]
各遺伝子の意味 = [使用する予測式の項数, どれだけ過去のデータを予測に用いるか,
                 予測に用いる銘柄番号, 活性化関数, 二項演算子, 各項の重み]
各遺伝子の型 = [int, int, int, int, int, activation_func[](numf_factors個),
               binary_operator[](num_factors個), int]
'''

class GeneticAlgorithmRecruiter:

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