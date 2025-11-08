"""
main02_1.py をベースに、JSAI2020論文 に基づく
累積利回り (CUMRET) の計算とプロット機能を追加したスクリプト。

主な変更点:
1. 論文 に合わせ、単純利回り(pct_change)から
   対数利回り(log return)を使用するように変更。
2. 論文 に基づき、テストデータでの検証ループ内で
   戦略リターン (sign(予測) * 実リターン) を記録。
3. ループ終了後、銘柄ごとに累積リターンを計算し、
   最後に銘柄間で平均して CUMRET (C_t^f) を算出。
4. MAE（平均絶対誤差）のグラフの代わりに、CUMRETの時系列グラフを
   プロットするよう変更。
5. (v2) Python 3.13 での ImportError に対応 (typing.list -> typing.List)
"""
import sys
import os
import random
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yfinance as yf
# --- Python 3.13 ImportError 修正 ---
from typing import List, Dict, Any  # 型ヒント(mypy準拠)のため
# -----------------------------------

# (sys.path.append は main02_1.py と同様のため省略)
sys.path.append(r'C:\Users\hi21yoshimura\.vscode\RedStoneWork\original_work')
sys.path.append(r'C:\Users\Y.Yoshimura\.vscode\GitHubRepository\RedStoneWork-1\original_work')

import tradercompany
from tradercompany.activation_funcs import identity, ReLU, sign, tanh
from tradercompany.binary_operators import add, diff, get_x, get_y, multiple, x_is_greater_than_y
from tradercompany.trader import Trader
from tradercompany.company import Company

# 乱数シードの固定
SEED = 2021
def fix_all_seeds(seed: int) -> None:
    """
    乱数シードを固定する関数。

    Args:
        seed (int): 固定するシード値
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

fix_all_seeds(SEED)

# --- 1. 実際の株価データを取得・加工 ---
print("--- yfinanceから株価データを取得・加工開始 ---")
tickers = ["7203.T", "6758.T"]
stock_names = ["TOYOTA", "SONY"]
data = yf.download(tickers, period="4y")

# --- JSAI2020 CUMRET: (変更) ---
# 論文 (2.1節) に従い、対数利回り r_i,t = log(X_i,t / X_i,t-1) を使用
df_y = np.log(data['Close'] / data['Close'].shift(1))
# -----------------------------------

df_y = df_y.dropna()
df_y.columns = stock_names
print("--- データ取得・加工完了 ---")
print(f"取得した対数利回りデータの先頭5行:")
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
recruit_methods = ["random", "genetic_algorithm"]
results_notuning = {}
results_tuning = {}
results_notuning_ma = {}
results_tuning_ma = {}

recruit_line = 0.6
num_traders = 400
mutation_rate = 0.01

# --- JSAI2020 CUMRET: (追加 & 修正) ---
# 戦略リターン (b_hat * r_t) を時系列で格納する辞書
# (np.ndarray は [TOYOTAリターン, SONYリターン] のような配列)
# --- Python 3.13 ImportError 修正 ---
strategy_returns_notuning: Dict[str, List[np.ndarray]] = {}
strategy_returns_tuning: Dict[str, List[np.ndarray]] = {}
# ---------------------------------


# --- 各採用手法で学習と検証を実行 ---
for method in recruit_methods:
    print(f"--- 採用手法 '{method}' で学習と検証を開始 ---")

    # モデルを構築
    model = Company(stock_names,
                    num_factors_max,
                    delay_time_max,
                    activation_funcs,
                    binary_operators,
                    num_traders=num_traders,
                    Q=recruit_line,
                    time_window=time_window,
                    how_recruit=method,
                    ga_mutation_rate=mutation_rate)

    # --- trainデータで学習 ---
    print("--- モデルの学習開始 ---")
    model.fit(df_y_train)
    print("--- モデルの学習完了 ---\n")

    # --- モデルの保存 ---
    model_filename = f"model_{method}.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)

    # --- 次の時刻の予測 ---
    prediction = model.aggregate()
    print(f"--- 時刻 {T_train} における次の時刻の予測値 ({method}) ---")
    print(prediction)
    print("------------------------------------------------\n")

    # --- JSAI2020 CUMRET: (追加) ---
    # この手法の結果を格納するリストを初期化
    strategy_returns_notuning[method] = []
    strategy_returns_tuning[method] = []
    # ---------------------------------

    # --- testデータに対する検証 (チューニングなし) ---
    print(f"--- テストデータでの検証開始（{method} - チューニングなし） ---")
    with open(model_filename, "rb") as f:
        model_no_tuning = pickle.load(f)

    current_errors_notuning = []
    # tqdmで進捗を表示
    for i, row in tqdm(df_y_test.iterrows(), total=len(df_y_test), desc=f"{method} no_tuning"):
        # 予測
        prediction_test = model_no_tuning.aggregate()
        
        # 誤差の計算 (従来通り)
        current_errors_notuning.append(np.abs(row.values - prediction_test))

        # --- JSAI2020 CUMRET: (追加) ---
        # 1. 予測値(prediction_test)からシグナル(b_hat)を決定
        signal = np.sign(prediction_test)
        # 2. シグナル * 実リターン(row.values) を計算
        strategy_return = signal * row.values 
        strategy_returns_notuning[method].append(strategy_return)
        # ---------------------------------

        # 新しいデータでモデルを更新
        model_no_tuning.fit_new_data(row.to_dict(), tuning=False)

    results_notuning[method] = np.array(current_errors_notuning)
    print("--- 検証完了 ---\n")


    # --- testデータに対する検証 (チューニングあり) ---
    print(f"--- テストデータでの検証開始（{method} - チューニングあり） ---")
    with open(model_filename, "rb") as f:
        model_tuning = pickle.load(f)

    current_errors_tuning = []
    for i, row in tqdm(df_y_test.iterrows(), total=len(df_y_test), desc=f"{method} tuning"):
        prediction_test = model_tuning.aggregate()
        
        # 誤差の計算 (従来通り)
        current_errors_tuning.append(np.abs(row.values - prediction_test))

        # --- JSAI2020 CUMRET: (追加) ---
        # 1. 予測値(prediction_test)からシグナル(b_hat)を決定
        signal = np.sign(prediction_test)
        # 2. シグナル * 実リターン(row.values) を計算
        strategy_return = signal * row.values
        strategy_returns_tuning[method].append(strategy_return)
        # ---------------------------------
        
        # 新しいデータでモデルを更新
        model_tuning.fit_new_data(row.to_dict(), tuning=True)

    results_tuning[method] = np.array(current_errors_tuning)
    print("--- 検証完了 ---\n")

# --- (従来の)精度の確認 (MAE) ---
days_ma = 5
for method in recruit_methods:
    results_notuning_ma[method] = pd.DataFrame(results_notuning[method], columns=stock_names, index=df_y_test.index).rolling(days_ma).mean()
    results_tuning_ma[method] = pd.DataFrame(results_tuning[method], columns=stock_names, index=df_y_test.index).rolling(days_ma).mean()

errors_baseline = np.abs(df_y_test.values - df_y_test.shift(1).values)
errors_baseline_ma = pd.DataFrame(errors_baseline, columns=stock_names, index=df_y_test.index).rolling(days_ma).mean()


# --- JSAI2020 CUMRET: (追加) ---
# --- 累積リターン(CUMRET C_t^f) の計算 ---
print("--- 累積リターン(CUMRET)の計算 ---")
# CUMRET (C_t^f) の時系列データを格納する辞書
cumret_results: Dict[str, pd.Series] = {} # <- 型ヒント修正
test_index = df_y_test.index

for method in recruit_methods:
    # --- No Tuning ---
    # 1. 戦略リターンのリストをDataFrameに変換
    df_strategy_ret_nt = pd.DataFrame(
        strategy_returns_notuning[method], 
        columns=stock_names, 
        index=test_index
    )
    # 2. 銘柄ごとの累積リターン (c_i,t^f) を計算
    df_cumret_per_stock_nt = df_strategy_ret_nt.cumsum()
    # 3. 銘柄間で平均し、CUMRET (C_t^f) を計算
    cumret_results[f"TC_{method}_notuning"] = df_cumret_per_stock_nt.mean(axis=1)

    # --- Tuning ---
    df_strategy_ret_t = pd.DataFrame(
        strategy_returns_tuning[method], 
        columns=stock_names, 
        index=test_index
    )
    df_cumret_per_stock_t = df_strategy_ret_t.cumsum()
    cumret_results[f"TC_{method}_tuning"] = df_cumret_per_stock_t.mean(axis=1)

# Baseline: Market (常に b_hat=1 としてホールド)
# (実リターン df_y_test をそのまま累積し、銘柄平均をとる)
df_cumret_market = df_y_test.cumsum()
cumret_results["Market"] = df_cumret_market.mean(axis=1)

print("--- 計算完了 ---\n")
# ---------------------------------


# --- JSAI2020 CUMRET: (変更) ---
# --- 累積リターン(CUMRET)のプロット ---
# (従来のMAEプロットの代わりに、CUMRETをプロットします)
print("--- 累積リターン(CUMRET)のプロット ---")
plt.figure(figsize=(12, 7)) # グラフサイズを調整

# 計算したCUMRETをすべてプロット
for label, cumret_series in cumret_results.items():
    # tuning あり/なし が分かりやすいよう、線種(linestyle)を変更
    linestyle = '--' if 'tuning' in label else '-'
    # 'Market' はグレーの点線にする
    if label == 'Market':
        linestyle = ':'
        color = 'gray'
        plt.plot(cumret_series, label=label, linestyle=linestyle, color=color)
    else:
        plt.plot(cumret_series, label=label, linestyle=linestyle)

# 論文の図1 に合わせて Y軸ラベル を設定
plt.title(f"Cumulative Return (CUMRET) (Q={recruit_line}, num_trader={num_traders}, mutation_rate={mutation_rate})")
plt.ylabel("Log Cumulative Return") #
plt.xlabel("Time")
plt.legend()       # 凡例を表示
plt.grid(True)     # グリッドを表示
plt.xticks(rotation=45, ha='right')
plt.tight_layout() # レイアウトを自動調整
plt.show()
print("---------------------\n")
# ---------------------------------


# --- 平均誤差の表示 (ここは変更なし) ---
print("--- 平均誤差 ---")
for i_stock, name in enumerate(stock_names):
    print(name)
    for method in recruit_methods:
        print(f"Trader-Company {method} notunig", results_notuning[method].T[i_stock].mean())
        print(f"Trader-Company {method} tuning", results_tuning[method].T[i_stock].mean())
    print("baseline", errors_baseline[i_stock].mean())
print("-----------------\n")

# --- モデルの解釈 (ここは変更なし) ---
print("--- モデルの解釈 (tuningありモデル) ---")
# (中身は main02_1.py と同様のため省略)
num_stock = len(stock_names)
# (エラー回避のため、ロードするモデルを明示的に指定)
with open(f"model_{recruit_methods[-1]}.pkl", "rb") as f:
    model_tuning = pickle.load(f)

traders_ranking_0 = np.argsort([trader.cumulative_error[0] for trader in model_tuning.traders]) 
traders_ranking_1 = np.argsort([trader.cumulative_error[1] for trader in model_tuning.traders]) 

print(stock_names[0])
print("Best trader's binary operators:", model_tuning.traders[traders_ranking_0[0]].binary_operator[0])
print("Best trader's activation functions:", model_tuning.traders[traders_ranking_0[0]].activation_func[1])
print("")
print(stock_names[1])
print("Best trader's binary operators:", model_tuning.traders[traders_ranking_1[0]].binary_operator[0])
print("Best trader's activation functions:", model_tuning.traders[traders_ranking_1[0]].activation_func[1])
print("---------------------\n")