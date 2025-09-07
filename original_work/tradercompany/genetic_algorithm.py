import numpy as np
import math
import random

'''
crossoverrate = 0.9 #交差率
mutation_rate = 0.01 #突然変異率
populations = 20    #個体群の大きさ
generations = 1     #世代
genoms = 8          #遺伝子の長さ

# --- Traderのパラメータ範囲 (Companyクラスから引用・仮定) ---
NUM_FACTORS_MAX = 4
DELAY_TIME_MAX = 2
NUM_STOCK = 2
ACTIVATION_FUNCS_COUNT = 4 # identity, ReLU, sign, tanh
BINARY_OPERATORS_COUNT = 8 # max, min, add, diff, multiple, get_x, get_y, x_is_greater_than_y


SEED = 2021
random.seed(SEED)
np.random.seed(SEED)


個体の遺伝子 = [num_factors, delay_P, delay_Q, stock_P, stock_Q,
               activation_func, binary_operator, w]
各遺伝子の意味 = [使用する予測式の項数, どれだけ過去のデータを予測に用いるか,
                 予測に用いる銘柄番号, 活性化関数, 二項演算子, 各項の重み]
各遺伝子の型 = [int, int, int, int, int, activation_func[](numf_factors個),
               binary_operator[](num_factors個), int]

'''


class GeneticAlgorithmRecruiter:
    """
    遺伝的アルゴリズムを用いて新しいTraderを生成（採用）するクラス。
    - 選択: ルーレット選択
    - 交叉: 一点交叉
    - 突然変異: 各パラメータを一定確率でランダムな値に変更
    """
    #mutation_rate = 0.01 #突然変異率

    SEED = 2021
    random.seed(SEED)
    np.random.seed(SEED)

    def __init__(self, company, mutation_rate):
        """
        Args:
            company (Company): Companyオブジェクト。Traderのパラメータ範囲などを取得するために使用。
            mutation_rate (float): 突然変異の発生確率。
        """
        self.company = company
        self.mutation_rate = mutation_rate

    def recruit(self, good_traders, num_to_recruit, i_stock):
        """
        新しいTraderのパラメータリストを生成する。

        Args:
            good_traders (list): 優秀な成績を残したTrader（親個体）のリスト。
            num_to_recruit (int): 生成する新しいTraderの数。
            i_stock (int): 対象となる株のインデックス。

        Returns:
            list: 新しく生成されたTraderのパラメータ（遺伝子）のリスト。
        """

        if not good_traders:
            # 優秀なTraderがいない場合は、ランダムなパラメータを返す
            return [self._generate_random_params() for _ in range(num_to_recruit)]

        # 適応度を計算 (エラーが小さいほど適応度が高い)
        fitness_scores = [1.0 / (trader.cumulative_error[i_stock] + 1e-6) for trader in good_traders]

        new_trader_params_list = []
        for _ in range(num_to_recruit):
            # 1. 選択 (ルーレット選択)
            parent1, parent2 = self._selection(good_traders, fitness_scores)

            # 2. 交叉 (一点交叉)
            child_params1, child_params2 = self.single_point_crossover(parent1, parent2, i_stock)

            # 3. 突然変異
            child_params1 = self._mutation(child_params1)
            child_params2 = self._mutation(child_params2)

            # 4. 解雇n人に対して親は2n人になるのでランダムで片方を雇用リストに追加
            #    ※ランダムじゃなくしたらオリジナリティがでるかも
            #    ※親の対象を上位Q%群からじゃなくて全個体にした方がいいかも？（疑似的にエリート主義になっている？）
            if np.random.rand() < 0.5:
                new_trader_params_list.append(child_params1)
            else:
                new_trader_params_list.append(child_params2)

        return new_trader_params_list

    def _selection(self, population, fitness_scores):
        """
        ルーレット選択により2つの親個体を選択する。
        """
        total_fitness = sum(fitness_scores)
        probabilities = [score / total_fitness for score in fitness_scores]

        # 確率分布に基づいて2つの親を重複ありで選択
        selected_indices = np.random.choice(len(population), size=2, p=probabilities)
        return population[selected_indices[0]], population[selected_indices[1]]

    def single_point_crossover(self, parent1, parent2, i_stock):
        """
        一点交叉により新しい子個体（パラメータ）を生成する。
        """
        params1 = parent1.get_params(i_stock)
        params2 = parent2.get_params(i_stock)

        # 交叉点を決定
        # num_factorsが異なる場合があるため、短い方の長さに合わせる
        size = min(len(params1), len(params2))
        crossover_point = random.randint(1, size - 1)

        # 新しい子個体のパラメータを作成
        child_params1 = params1[:crossover_point] + params2[crossover_point:]
        child_params2 = params2[:crossover_point] + params1[crossover_point:]

        # num_factorの値を実際の要素数に合わせる
        child_params1[0]['num_factor'] = len(child_params1) - 1
        child_params2[0]['num_factor'] = len(child_params2) - 1

        return child_params1, child_params2


    #交叉
    def two_point_crossover(self, parent1, parent2, i_stock):
        '''交叉の関数(二点交叉)
            input: 混ぜ合わせたい個体のペア
            output: 交叉後の個体のペア'''

        params1 = parent1.get_params(i_stock)
        params2 = parent2.get_params(i_stock)
        size = min(len(params1), len(params2))
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


    def _mutation(self, params):
        """
        各パラメータを一定確率で突然変異させる。
        """
        # num_factorの突然変異
        if random.random() < self.mutation_rate:
            new_num_factors = random.randint(1, self.company.num_factors_max)
            # パラメータリストの長さを調整
            current_len = len(params) - 1
            if new_num_factors > current_len:
                # 要素を追加
                for _ in range(new_num_factors - current_len):
                    params.append(self._generate_random_factor_params())
            elif new_num_factors < current_len:
                # 要素を削除
                params = params[:new_num_factors + 1]
            params[0]['num_factor'] = new_num_factors

        # 各ファクターのパラメータの突然変異
        for i in range(1, len(params)):
            if random.random() < self.mutation_rate:
                params[i] = self._generate_random_factor_params()

        return params

    def _generate_random_params(self):
        """
        Traderの完全なランダムパラメータを生成する。
        """
        num_factor = random.randint(1, self.company.num_factors_max)
        params = [{'num_factor': num_factor}]
        for _ in range(num_factor):
            params.append(self._generate_random_factor_params())
        return params

    def _generate_random_factor_params(self):
        """
        1つのファクターに対するランダムなパラメータを生成する。
        """
        return {
            "delay_P": random.randint(0, self.company.delay_time_max - 1),
            "delay_Q": random.randint(0, self.company.delay_time_max - 1),
            "stock_P": random.randint(0, self.company.num_stock - 1),
            "stock_Q": random.randint(0, self.company.num_stock - 1),
            "activation_func": random.randint(0, len(self.company.activation_funcs) - 1),
            "binary_operator": random.randint(0, len(self.company.binary_operators) - 1),
        }
