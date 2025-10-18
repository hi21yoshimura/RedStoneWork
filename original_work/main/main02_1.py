import sys
import os
import random
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yfinance as yf

# tradercompanyライブラリがインストールされているパスを追加
# 必要に応じて変更してください
# sys.path.append('../')
sys.path.append(r'C:\Users\hi21yoshimura\.vscode\RedStoneWork\original_work')
#sys.path.append(r'C:\Users\Y.Yoshimura\.vscode\GitHubRepository\RedStoneWork\RedStoneWork\original_work')
sys.path.append(r'C:\Users\Y.Yoshimura\.vscode\GitHubRepository\RedStoneWork-1\original_work')

# Gemini----------------------------------------------------------------
# この部分をまるごと置き換えてください
# main01.py があるディレクトリの絶対パスを取得
# current_dir = os.path.dirname(os.path.abspath(__file__))
# 1つ上の階層（RedStoneWork）に移動し、tradercompany-main フォルダへのパスを作成
# tradercompany_path = os.path.join(os.path.dirname(current_dir), 'tradercompany-main')
# Pythonがライブラリを探す場所のリストに、上記パスを追加
# sys.path.append(tradercompany_path)
# ----------------------------------------------------------------------

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

# --- 1. 実際の株価データを取得・加工 ---
print("--- yfinanceから株価データを取得・加工開始 ---")
tickers = ["7203.T", "6758.T"]
stock_names = ["TOYOTA", "SONY"]
data = yf.download(tickers, period="2y")
df_y = data['Close'].pct_change()
df_y = df_y.dropna()
df_y.columns = stock_names
print("--- データ取得・加工完了 ---")
print("取得したデータの先頭5行:")
print(df_y.head())
print("--------------------------\n")

# --- Trader-Company method ---
activation_funcs = [identity, ReLU, sign, tanh]
binary_operators = [max, min, add, diff, multiple, get_x, get_y, x_is_greater_than_y]
time_window = 200
delay_time_max = 2
num_factors_max = 4

# --- trainとtestに分ける ---
T_train = 250
df_y_train = df_y.iloc[:T_train, :]
df_y_test = df_y.iloc[T_train:, :]

# --- 比較したい採用手法をリストで指定 ---
recruit_methods = ["random", "genetic_algorithm"] # <--- 変更点: 比較したい手法をリスト化
results_notuning = {}
results_tuning = {}
results_notuning_ma = {}
results_tuning_ma = {}

# --- 各採用手法で学習と検証を実行 ---
for method in recruit_methods:
    print(f"--- 採用手法 '{method}' で学習と検証を開始 ---")

    # モデルを構築
    model = Company(stock_names,
                    num_factors_max,
                    delay_time_max,
                    activation_funcs,
                    binary_operators,
                    num_traders=40,
                    Q=0.2,
                    time_window=time_window,
                    how_recruit=method) # <--- 変更点: 手法を指定

    # --- trainデータで学習 ---
    print("--- モデルの学習開始 ---")
    model.fit(df_y_train)
    print("--- モデルの学習完了 ---\n")

    # --- モデルの保存 ---
    model_filename = f"model_{method}.pkl" # 手法ごとにファイル名を変える
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)

    # --- 次の時刻の予測 ---
    prediction = model.aggregate()
    print(f"--- 時刻 {T_train} における次の時刻の予測値 ({method}) ---")
    print(prediction)
    print("------------------------------------------------\n")


    # --- testデータに対する検証 (チューニングなし) ---
    print(f"--- テストデータでの検証開始（{method} - チューニングなし） ---")
    with open(model_filename, "rb") as f:
        model_no_tuning = pickle.load(f)

    current_errors_notuning = []
    for i, row in tqdm(df_y_test.iterrows(), total=len(df_y_test), desc=f"{method} no_tuning"):
        prediction_test = model_no_tuning.aggregate()
        current_errors_notuning.append(np.abs(row.values - prediction_test))
        model_no_tuning.fit_new_data(row.to_dict(), tuning=False)

    results_notuning[method] = np.array(current_errors_notuning) # 手法名をキーにして結果を保存
    print("--- 検証完了 ---\n")


    # --- testデータに対する検証 (チューニングあり) ---
    print(f"--- テストデータでの検証開始（{method} - チューニングあり） ---")
    with open(model_filename, "rb") as f:
        model_tuning = pickle.load(f)

    current_errors_tuning = []
    for i, row in tqdm(df_y_test.iterrows(), total=len(df_y_test), desc=f"{method} tuning"):
        prediction_test = model_tuning.aggregate()
        current_errors_tuning.append(np.abs(row.values - prediction_test))
        model_tuning.fit_new_data(row.to_dict(), tuning=True)

    results_tuning[method] = np.array(current_errors_tuning) # 手法名をキーにして結果を保存
    print("--- 検証完了 ---\n")

# --- 精度の確認 ---
days_ma = 5

# 移動平均を計算
for method in recruit_methods:
    results_notuning_ma[method] = pd.DataFrame(results_notuning[method], columns=stock_names, index=df_y_test.index).rolling(days_ma).mean()
    results_tuning_ma[method] = pd.DataFrame(results_tuning[method], columns=stock_names, index=df_y_test.index).rolling(days_ma).mean()

# baseline method
errors_baseline = np.abs(df_y_test.values - df_y_test.shift(1).values)
errors_baseline_ma = pd.DataFrame(errors_baseline, columns=stock_names, index=df_y_test.index).rolling(days_ma).mean()

# --- 精度のプロット ---
print("--- 精度のプロット ---")
for i_stock, name in enumerate(stock_names):
    print(name)
    plt.figure() # <--- 変更点: 銘柄ごとに新しい図を作成
    # 各手法の結果をプロット
    for method in recruit_methods:
        plt.plot(results_notuning_ma[method].iloc[:, i_stock], label=f"TC_{method}_notuning")
        #plt.plot(results_tuning_ma[method].iloc[:, i_stock], label=f"TC_{method}_tuning")

    #plt.plot(errors_baseline_ma.iloc[:, i_stock], label="baseline")
    plt.xlabel("time")
    plt.ylabel("mean average error")
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
print("---------------------\n")


# --- 平均誤差の表示 ---
print("--- 平均誤差 ---")
for i_stock, name in enumerate(stock_names):
    print(name)
    for method in recruit_methods:
        print(f"Trader-Company {method} notunig", results_notuning[method].T[i_stock].mean())
        print(f"Trader-Company {method} tuning", results_tuning[method].T[i_stock].mean())
    print("baseline", errors_baseline[i_stock].mean())
print("-----------------\n")

# --- モデルの解釈 (例として最後のtuningありモデルを使用) ---
print("--- モデルの解釈 (tuningありモデル) ---")
# 注意: この部分は最後に実行された手法(リストの最後の要素)のモデルを解釈します。
#      特定のモデルを解釈したい場合は、該当する .pkl ファイルをロードしてください。
#      例: method_to_analyze = "gmm"
#          with open(f"model_{method_to_analyze}.pkl", "rb") as f:
#              model_for_analysis = pickle.load(f)
#          traders_ranking_0 = np.argsort([trader.cumulative_error[0] for trader in model_for_analysis.traders])
#          ... 以下同様 ...

num_stock = len(stock_names)
traders_ranking_0 = np.argsort([trader.cumulative_error[0] for trader in model_tuning.traders]) # 最後に実行された tuning ありモデルを使用
traders_ranking_1 = np.argsort([trader.cumulative_error[1] for trader in model_tuning.traders]) # 最後に実行された tuning ありモデルを使用

print(stock_names[0])
print("Best trader's binary operators:", model_tuning.traders[traders_ranking_0[0]].binary_operator[0])
print("Best trader's activation functions:", model_tuning.traders[traders_ranking_0[0]].activation_func[1])
print("")
print(stock_names[1])
print("Best trader's binary operators:", model_tuning.traders[traders_ranking_1[0]].binary_operator[0])
print("Best trader's activation functions:", model_tuning.traders[traders_ranking_1[0]].activation_func[1])
print("---------------------\n")