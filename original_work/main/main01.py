# tutorial01.py

import sys
import os
import random
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# tradercompanyライブラリがインストールされているパスを追加
# 必要に応じて変更してください
#sys.path.append('../')
sys.path.append(r'C:\Users\Y.Yoshimura\.vscode\GitHubRepository\RedStoneWork\RedStoneWork\original_work')

"""
# Gemini----------------------------------------------------------------
# この部分をまるごと置き換えてください

# main01.py があるディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
# 1つ上の階層（RedStoneWork）に移動し、tradercompany-main フォルダへのパスを作成
tradercompany_path = os.path.join(os.path.dirname(current_dir), 'tradercompany-main')
# Pythonがライブラリを探す場所のリストに、上記パスを追加
sys.path.append(tradercompany_path)
# ----------------------------------------------------------------------
"""

import tradercompany
from tradercompany.activation_funcs import identity, ReLU, sign, tanh
from tradercompany.binary_operators import add, diff, get_x, get_y, multiple, x_is_greater_than_y
from tradercompany.trader import Trader
from tradercompany.company import Company

# 乱数シードの固定
SEED = 2021
def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

fix_all_seeds(SEED)

# --- デモデータの作成 ---
def create_dataset(T, sigma_t):
    def simulation(y_t, sigma):
        y_t1 = np.zeros(2)
        y_t1[0] = 1.0*tanh(y_t[0]) + 0.8*y_t[0]*y_t[1] + 1.0*y_t[1] - 1.0*ReLU(min(y_t[0], y_t[1])) + sigma*np.random.randn()
        y_t1[1] = +0.6*sign(y_t[1]) + 0.5*y_t[0]*y_t[1] - 1.0*max(y_t[0], y_t[1]) + sigma*np.random.randn()
        return y_t1
    
    y = np.zeros((2, T))
    y_without_noise = np.zeros((2, T))
    y[:,0] = np.array([0.1, 0.1])
    y_without_noise[:,0] = np.array([0.1, 0.1])
    
    for t in range(1, T):
        y[:,t] = simulation(y[:,t-1], sigma_t)
        y_without_noise[:,t] = simulation(y[:,t-1], 0.0)
    
    plt.plot(y[0], color = "#cc0000", label = "stock0")
    plt.plot(y[1], color = "#083090", label = "stock1")
    plt.plot(y_without_noise[0], color = "#cc0000", linestyle = "--", label = "stock0" + "(w/o noise)")
    plt.plot(y_without_noise[1], color = "#083090", linestyle = "--", label = "stock1" + "(w/o noise)")
    plt.xlabel("time", fontsize = 18)
    plt.ylabel("y", fontsize = 18)
    plt.xlim([T-100, T])
    plt.legend()
    plt.show()
    plt.close()
    
    return y, y_without_noise

sigma = 0.1
T_total = 500
y, y_without_noise = create_dataset(T_total, sigma)

df_y = pd.DataFrame(y, index=["stock0", "stock1"]).T
print("--- 作成されたデータの先頭5行 ---")
print(df_y.head())
print("----------------------------------\n")

# --- Trader-Company method ---
activation_funcs = [identity, ReLU, sign, tanh]
binary_operators = [max, min, add, diff, multiple, get_x, get_y, x_is_greater_than_y]

stock_names = ["stock0", "stock1"]
time_window = 200
delay_time_max = 2
num_factors_max = 4

model = Company(stock_names, 
                num_factors_max, 
                delay_time_max, 
                activation_funcs, 
                binary_operators, 
                num_traders=40, 
                Q=0.2, 
                time_window=time_window, 
                how_recruit="random")

# --- trainとtestに分ける ---
T_train = 400
df_y_train = df_y.iloc[:T_train, :]
df_y_test = df_y.iloc[T_train:, :]

# --- trainデータで学習 ---
print("--- モデルの学習開始 ---")
model.fit(df_y_train)
print("--- モデルの学習完了 ---\n")

# --- モデルの保存 ---
# 予測する際にデータを追加していくようになっているの学習後の状態を保存しておく。
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# --- 次の時刻の予測 ---
# 時刻t+1の予測
prediction = model.aggregate()
print(f"--- 時刻 {T_train} における次の時刻の予測値 ---")
print(prediction)
print("------------------------------------------------\n")

#"""
# --- testデータに対する検証 ---
print("--- テストデータでの検証開始（チューニングなし） ---")
with open("model.pkl", "rb") as f:
    model_no_tuning = pickle.load(f)

errors_test_notuning = []
for i, row in tqdm(df_y_test.iterrows(), total=len(df_y_test)):
    prediction_test = model_no_tuning.aggregate()
    errors_test_notuning.append(np.abs(row.values - prediction_test))
    
    # tuning==Falseの場合、データが追加されても重みの更新などパラメータは変わらない
    model_no_tuning.fit_new_data(row.to_dict(), tuning=False)
print("--- 検証完了 ---\n")


print("--- テストデータでの検証開始（チューニングあり） ---")
with open("model.pkl", "rb") as f:
    model_tuning = pickle.load(f)

errors_test_tuning = []
for i, row in tqdm(df_y_test.iterrows(), total=len(df_y_test)):
    prediction_test = model_tuning.aggregate()
    errors_test_tuning.append(np.abs(row.values - prediction_test))
    
    # tuning==Trueの場合、データが追加された際に重みの更新などパラメータが調整される
    model_tuning.fit_new_data(row.to_dict(), tuning=True)
print("--- 検証完了 ---\n")

#"""

# --- 精度の確認 ---
days_ma = 5

# trader-company errors with no-tuning
errors_test_notuning = np.array(errors_test_notuning)
errors_test_notuning_ma = pd.DataFrame(errors_test_notuning).rolling(days_ma).mean()

# trader-company errors with tuning
errors_test_tuning = np.array(errors_test_tuning)
errors_test_tuning_ma = pd.DataFrame(errors_test_tuning).rolling(days_ma).mean()

# baseline method
errors_baseline = np.abs(y[:,T_train+1:] - y[:,T_train:-1])
errors_baseline_ma = pd.DataFrame(errors_baseline.T).rolling(days_ma).mean()

# lower bound
errors_lower_bound = np.abs(y[:,T_train+1:] - y_without_noise[:,T_train+1:])
errors_lower_bound_ma = pd.DataFrame(errors_lower_bound.T).rolling(days_ma).mean()

# --- 精度のプロット ---
print("--- 精度のプロット ---")
for i_stock, name in enumerate(stock_names):
    print(name)
    plt.plot(errors_test_notuning_ma[i_stock], label="trader-company_notuning")
    plt.plot(errors_test_tuning_ma[i_stock], label="trader-company_tuning")
    plt.plot(errors_baseline_ma[i_stock], label="baseline")
    plt.plot(errors_lower_bound_ma[i_stock], label="lower-bound")
    plt.xlabel("time")
    plt.ylabel("mean average error")
    plt.legend()
    plt.show()
print("---------------------\n")


# --- 平均誤差の表示 ---
print("--- 平均誤差 ---")
for i_stock, name in enumerate(stock_names):
    print(name)
    print("Trader-Company notunig", errors_test_notuning.T[i_stock].mean())
    print("Trader-Company tuning", errors_test_tuning.T[i_stock].mean())
    print("baseline", errors_baseline[i_stock].mean())
    print("lower bound", errors_lower_bound[i_stock].mean())
print("-----------------\n")

# --- モデルの解釈 ---
print("--- モデルの解釈 ---")
num_stock = len(stock_names)
best_trader_for = [[], []]

traders_ranking_0 = np.argsort([trader.cumulative_error[0] for trader in model_tuning.traders])
traders_ranking_1 = np.argsort([trader.cumulative_error[1] for trader in model_tuning.traders])

# best trader for stock0
print(stock_names[0])
print("Best trader's binary operators:", model_tuning.traders[traders_ranking_0[0]].binary_operator[0])
print("Best trader's activation functions:", model_tuning.traders[traders_ranking_0[0]].activation_func[1])
print("")
# best trader for stock1
print(stock_names[1])
print("Best trader's binary operators:", model_tuning.traders[traders_ranking_1[0]].binary_operator[0])
print("Best trader's activation functions:", model_tuning.traders[traders_ranking_1[0]].activation_func[1])
print("---------------------\n")