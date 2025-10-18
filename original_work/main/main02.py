# main02.py

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
#sys.path.append('../')
sys.path.append(r'C:\Users\hi21yoshimura\.vscode\RedStoneWork\original_work')
sys.path.append(r'C:\Users\Y.Yoshimura\.vscode\GitHubRepository\RedStoneWork-1\original_work')

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

# --- 1. 実際の株価データを取得・加工 ---
print("--- yfinanceから株価データを取得・加工開始 ---")

# 取得したい銘柄のティッカーシンボルを指定
# (例: トヨタ自動車、ソニーグループ)
tickers = ["7203.T", "6758.T"]
stock_names = ["TOYOTA", "SONY"]

# データの取得期間を1年間に設定
data = yf.download(tickers, period="4y")

# 'Adj Close'(調整後終値)のみを使用し、日次収益率（前日比の変動率）に変換
# Trader-Company法は価格そのものではなく、収益率を予測するためこの変換が重要
df_y = data['Close'].pct_change()

# 欠損値（NaN）を含む最初の行を削除
df_y = df_y.dropna()

# カラム名を分かりやすい名前に変更
df_y.columns = stock_names

print("--- データ取得・加工完了 ---")
print("取得したデータの先頭5行:")
print(df_y.head())
print("--------------------------\n")

# --- Trader-Company method ---
activation_funcs = [identity, ReLU, sign, tanh]
binary_operators = [max, min, add, diff, multiple, get_x, get_y, x_is_greater_than_y]

#stock_names = ["stock0", "stock1"]
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
                how_recruit="genetic_algorithm")

# --- trainとtestに分ける ---
T_train = 500
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
errors_test_notuning_ma = pd.DataFrame(errors_test_notuning, columns=stock_names, index=df_y_test.index).rolling(days_ma).mean()

# trader-company errors with tuning
errors_test_tuning = np.array(errors_test_tuning)
errors_test_tuning_ma = pd.DataFrame(errors_test_tuning, columns=stock_names, index=df_y_test.index).rolling(days_ma).mean()

# baseline method
# df_y_test の値を1日ずらして、今日の値と昨日の値の差を計算
errors_baseline = np.abs(df_y_test.values - df_y_test.shift(1).values)
errors_baseline_ma = pd.DataFrame(errors_baseline, columns=stock_names, index=df_y_test.index).rolling(days_ma).mean()

# lower bound
# デモデータと違ってノイズのない真の値が存在せず、本来の値とノイズの分離ができないから出力しない
#errors_lower_bound = np.abs(y[:,T_train+1:] - y_without_noise[:,T_train+1:])
#errors_lower_bound_ma = pd.DataFrame(errors_lower_bound.T).rolling(days_ma).mean()

# --- 精度のプロット ---
print("--- 精度のプロット ---")
for i_stock, name in enumerate(stock_names):
    print(name)
    plt.plot(errors_baseline_ma.iloc[:, i_stock], label="baseline")
    plt.plot(errors_test_notuning_ma.iloc[:, i_stock], label="trader-company_notuning")
    #plt.plot(errors_test_tuning_ma.iloc[:, i_stock], label="trader-company_tuning")
    #plt.plot(errors_lower_bound_ma.iloc[:, i_stock], label="lower-bound")
    plt.xlabel("time")
    plt.ylabel("mean average error")
    plt.legend()

    plt.xticks(rotation=45, ha='right') # ラベルを45度回転
    plt.tight_layout() # レイアウトを自動調整

    plt.show()
print("---------------------\n")


# --- 平均誤差の表示 ---
print("--- 平均誤差 ---")
for i_stock, name in enumerate(stock_names):
    print(name)
    print("Trader-Company notunig", errors_test_notuning.T[i_stock].mean())
    print("Trader-Company tuning", errors_test_tuning.T[i_stock].mean())
    print("baseline", errors_baseline[i_stock].mean())
    #print("lower bound", errors_lower_bound[i_stock].mean())
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