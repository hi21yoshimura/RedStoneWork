"""
Trader-Company法 卒業研究分析スクリプト (Final Version 8)

修正点:
- BASE_PKL_DIR, CACHE_FILE のパス更新
- (A)最終累積利回りをRandom/Geneticの比較形式(2枚並び)に変更
- ヒートマップの配色を「赤(低) -> 白 -> 緑(高)」にカスタム設定
- (D)のタイトル変更と全タイトルの文字サイズ拡大
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from tqdm import tqdm
import yfinance as yf

# ---------------------------------------------------------
# フォント設定 (日本語対応)
# ---------------------------------------------------------
def configure_fonts():
    """OSに合わせて日本語フォントを自動設定"""
    import platform
    system = platform.system()
    
    if system == 'Windows':
        font_family = 'Meiryo'
    elif system == 'Darwin': # Mac
        font_family = 'Hiragino Sans'
    else: # Linux etc
        font_family = 'TakaoGothic'
        
    plt.rcParams['font.family'] = font_family
    plt.rcParams['axes.unicode_minus'] = False 
    
configure_fonts()

# ---------------------------------------------------------
# 設定セクション
# ---------------------------------------------------------

# 指定されたパス設定
BASE_PKL_DIR = r"C:\Users\Y.Yoshimura\.vscode\GitHubRepository\RedStoneWork-1\original_work\model\pkl"
CACHE_FILE = "final_experiment_results_v4.csv"

# パラメータ設定
DELAY_PARAMS = [1, 2, 7, 30]
FACTOR_PARAMS = [2, 4, 10, 15]

# ヒートマップ共通設定
HEATMAP_VMIN = -0.3
HEATMAP_VMAX = 0.3
TITLE_FONT_SIZE = 18  # タイトルの文字サイズ

# 株価データ取得設定
START_DATE = '2017-01-01'
END_DATE = '2021-01-01'
T_TRAIN = 500
SEED = 2021

# 銘柄リスト
TICKERS = [
    "1332.T", "2802.T", "3402.T", "3407.T", "4502.T",
    "4901.T", "5108.T", "5401.T", "6301.T", "6501.T",
    "6503.T", "6752.T", "6758.T", "7203.T", "7267.T",
    "7751.T", "8001.T", "8031.T", "8058.T", "8801.T"
]
STOCK_NAMES = [
    "Nissui", "Ajinomoto", "Toray", "Asahi Kasei", "Takeda",
    "Fujifilm", "Bridgestone", "Nippon Steel", "Komatsu", "Hitachi",
    "Mitsubishi Elec", "Panasonic", "Sony", "Toyota", "Honda",
    "Canon", "Itochu", "Mitsui", "Mitsubishi Corp", "Mitsui Fudosan"
]

# システムパス設定
sys.path.append(os.getcwd())

# ---------------------------------------------------------
# 関数定義
# ---------------------------------------------------------

def fix_all_seeds(seed):
    import random
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_stock_data():
    """株価データを取得・加工してテストデータ(df_y_test)を返す"""
    print("--- 株価データを取得中... ---")
    data = yf.download(TICKERS, START_DATE, END_DATE, progress=False)
    
    # 対数利回り計算
    df_y = np.log(data['Close'] / data['Close'].shift(1))
    
    # クリーニング
    df_y = df_y.replace([np.inf, -np.inf], np.nan)
    df_y = df_y.dropna(axis=0, how='all')
    df_y = df_y.fillna(0)
    df_y.columns = STOCK_NAMES
    
    # Train/Test分割
    df_y_test = df_y.iloc[T_TRAIN:, :]
    return df_y_test

def get_model_paths(file_num):
    """ファイル番号に基づきモデルパスを生成"""
    dir_path = os.path.join(BASE_PKL_DIR, str(file_num))
    path_random = os.path.join(dir_path, "model_random.pkl")
    path_genetic = os.path.join(dir_path, "model_genetic_algorithm.pkl")
    return path_random, path_genetic

def calculate_rr(daily_returns_df):
    """リスク・リターン比の計算 (平均 / 標準偏差)"""
    daily_mean = daily_returns_df.mean(axis=1)
    if daily_mean.std() == 0:
        return 0
    return daily_mean.mean() / daily_mean.std()

def run_backtest_with_loaded_model(model_path, df_y_test, desc="Backtest"):
    """モデルを読み込みバックテストを実行 (tqdm付き)"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    strategy_returns = []
    
    for index, row in tqdm(df_y_test.iterrows(), total=len(df_y_test), desc=desc, leave=False):
        prediction = model.aggregate()
        signal = np.sign(prediction)
        daily_return = signal * row.values
        strategy_returns.append(daily_return)
        model.fit_new_data(row.to_dict(), tuning=True)
        
    df_returns = pd.DataFrame(strategy_returns, index=df_y_test.index, columns=STOCK_NAMES)
    
    cum_returns = df_returns.cumsum().mean(axis=1)
    final_cumret = cum_returns.iloc[-1]
    
    rr = calculate_rr(df_returns)
    
    return cum_returns, final_cumret, rr

def process_experiments():
    """全実験を実行 (全体進捗tqdm付き)"""
    fix_all_seeds(SEED)
    df_y_test = get_stock_data()
    
    market_cum = df_y_test.cumsum().mean(axis=1)
    final_market_cum = market_cum.iloc[-1]
    market_rr = calculate_rr(df_y_test)
    
    min_market_idx = market_cum.idxmin()
    val_market_min = market_cum[min_market_idx]
    
    print(f"Market Stats: CUMRET={final_market_cum:.4f}, R/R={market_rr:.4f}")

    results = []
    
    experiment_params = []
    file_num = 1
    for delay in DELAY_PARAMS:
        for factors in FACTOR_PARAMS:
            experiment_params.append({
                "file_num": file_num,
                "delay": delay,
                "factors": factors
            })
            file_num += 1

    print("\n--- 実験開始 (全16条件 × 2モデル) ---")
    
    for param in tqdm(experiment_params, desc="Total Progress"):
        file_num = param["file_num"]
        delay = param["delay"]
        factors = param["factors"]
        
        path_random, path_genetic = get_model_paths(file_num)
        
        try:
            # Random Backtest
            cum_rand, final_rand, rr_rand = run_backtest_with_loaded_model(
                path_random, df_y_test, desc=f"No.{file_num} Random"
            )
            
            # Genetic Backtest
            cum_gen, final_gen, rr_gen = run_backtest_with_loaded_model(
                path_genetic, df_y_test, desc=f"No.{file_num} Genetic"
            )
            
            # 乖離計算
            diff_rand = abs(cum_rand[min_market_idx] - val_market_min)
            diff_gen = abs(cum_gen[min_market_idx] - val_market_min)

            results.append({
                "file_num": file_num,
                "delay_time_max": delay,
                "num_factors_max": factors,
                "cumret_random": final_rand,
                "rr_random": rr_rand,
                "diff_min_random": diff_rand,
                "cumret_genetic": final_gen,
                "rr_genetic": rr_gen,
                "diff_min_genetic": diff_gen
            })

        except Exception as e:
            tqdm.write(f"  [Error] No.{file_num}: {e}")

    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results.to_csv(CACHE_FILE, index=False)
        print(f"\nデータを {CACHE_FILE} に保存しました。")
        
    return df_results, (final_market_cum, market_rr)

def show_average_table(df, market_stats):
    """全16モデルの平均値を計算してテーブル表示"""
    market_cum, market_rr = market_stats

    avg_cumret_random = df["cumret_random"].mean()
    avg_rr_random = df["rr_random"].mean()
    
    avg_cumret_genetic = df["cumret_genetic"].mean()
    avg_rr_genetic = df["rr_genetic"].mean()

    table_data = {
        "Method": ["Market", "TC_random_tuning (Avg)", "TC_genetic_algorithm_tuning (Avg)"],
        "CUMRET (Mean)": [market_cum, avg_cumret_random, avg_cumret_genetic],
        "R/R (Mean)": [market_rr, avg_rr_random, avg_rr_genetic]
    }
    
    df_table = pd.DataFrame(table_data)
    df_table.set_index("Method", inplace=True)
    
    print("\n【全16モデルの平均パフォーマンス比較】")
    print(df_table.to_markdown())
    df_table.to_csv("average_comparison_table.csv")

def plot_heatmaps(df):
    """ヒートマップを出力 (要望に合わせて構成変更)"""
    print("\n--- ヒートマップを作成中 ---")
    
    # カスタムカラーマップ作成: 赤(低) -> 白(中) -> 緑(高)
    # 値が -0.3(赤) ... 0(白) ... 0.3(緑) となるように
    colors = ["red", "white", "green"]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("RedWhiteGreen", colors)

    heatmap_kwargs = {
        'annot': True, 
        'fmt': ".3f", 
        'cbar': True,
        'vmin': HEATMAP_VMIN,
        'vmax': HEATMAP_VMAX,
        'cmap': custom_cmap # カスタムカラーを使用
    }
    
    # (D) のデータを計算: Genetic乖離 - Random乖離
    df["diff_improvement"] = df["diff_min_genetic"] - df["diff_min_random"]

    # --- 1. 画像(A): 最終累積利回り (Random & Genetic) ---
    fig_a, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Random
    pivot_cum_rand = df.pivot(index="delay_time_max", columns="num_factors_max", values="cumret_random")
    pivot_cum_rand = pivot_cum_rand.sort_index(ascending=False)
    sns.heatmap(pivot_cum_rand, ax=ax1, **heatmap_kwargs)
    ax1.set_title("最終累積利回り（Random）", fontsize=TITLE_FONT_SIZE)
    ax1.set_xlabel("Factors")
    ax1.set_ylabel("Delay")

    # Genetic
    pivot_cum_gen = df.pivot(index="delay_time_max", columns="num_factors_max", values="cumret_genetic")
    pivot_cum_gen = pivot_cum_gen.sort_index(ascending=False)
    sns.heatmap(pivot_cum_gen, ax=ax2, **heatmap_kwargs)
    ax2.set_title("最終累積利回り（Genetic algorithm）", fontsize=TITLE_FONT_SIZE)
    ax2.set_xlabel("Factors")
    ax2.set_ylabel("")
    ax2.set_yticks([])

    plt.tight_layout()
    plt.savefig("heatmap_cumret_comparison.png")
    print("画像保存: heatmap_cumret_comparison.png")
    plt.close()

    # --- 2. 画像(B)&(C): 乖離の比較 (Market最低時) ---
    fig_bc, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 配色は「0に近いほど良い(白)」など解釈が異なる場合もありますが、
    # ここでは統一感を出すため同じRed-White-Greenを使うか、
    # あるいは乖離なので「白(0)が良い」ことを示すために
    # 'Reds'や'Blues'を使うのが一般的ですが、要望により統一しますか？
    # 前回のコードではReds/Bluesを使っていました。
    # 今回の指示は「(B),(C)のように繋げて」「色は高い方から緑...」は(A)に対する指示と解釈できます。
    # ですが「すべてのヒートマップのカラー範囲を...」という以前の指示もあるので、
    # 乖離についても、今回は視認性を優先し、以前のReds/Blues構成を維持します。
    # (乖離は「小さい方が良い」ので、緑-白-赤だと混乱を招く可能性があります)
    
    heatmap_kwargs_bc = heatmap_kwargs.copy()
    del heatmap_kwargs_bc['cmap'] # デフォルトまたは指定色に戻す

    # (B) Random 乖離
    pivot_diff_rand = df.pivot(index="delay_time_max", columns="num_factors_max", values="diff_min_random")
    pivot_diff_rand = pivot_diff_rand.sort_index(ascending=False)
    sns.heatmap(pivot_diff_rand, cmap="Reds", ax=ax3, **heatmap_kwargs_bc)
    ax3.set_title("(B) Market最低時の乖離 (Random)", fontsize=TITLE_FONT_SIZE)
    ax3.set_xlabel("Factors")
    ax3.set_ylabel("Delay")

    # (C) Genetic 乖離
    pivot_diff_gen = df.pivot(index="delay_time_max", columns="num_factors_max", values="diff_min_genetic")
    pivot_diff_gen = pivot_diff_gen.sort_index(ascending=False)
    sns.heatmap(pivot_diff_gen, cmap="Blues", ax=ax4, **heatmap_kwargs_bc)
    ax4.set_title("(C) Market最低時の乖離 (Genetic)", fontsize=TITLE_FONT_SIZE)
    ax4.set_xlabel("Factors")
    ax4.set_ylabel("")
    ax4.set_yticks([])

    plt.tight_layout()
    plt.savefig("heatmap_deviation_BC.png")
    print("画像保存: heatmap_deviation_BC.png")
    plt.close()

    # --- 3. 画像(D): Market最低時の累積利回りの差 ---
    plt.figure(figsize=(8, 6))
    pivot_diff_imp = df.pivot(index="delay_time_max", columns="num_factors_max", values="diff_improvement")
    pivot_diff_imp = pivot_diff_imp.sort_index(ascending=False)
    
    # 差分は RdBu (青=High=Genetic乖離大, 赤=Low=Random乖離大)
    # ここは以前の指定「高い方から青、赤」を維持
    sns.heatmap(pivot_diff_imp, cmap="RdBu", center=0, 
                annot=True, fmt=".3f", cbar=True, vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX)
    
    plt.title("Market最低時の累積利回りの差", fontsize=TITLE_FONT_SIZE)
    plt.xlabel("Factors")
    plt.ylabel("Delay")
    plt.tight_layout()
    plt.savefig("heatmap_deviation_diff.png")
    print("画像保存: heatmap_deviation_diff.png")
    plt.close()

# ---------------------------------------------------------
# メイン実行部
# ---------------------------------------------------------

if __name__ == "__main__":
    if os.path.exists(CACHE_FILE):
        print(f"既存のデータ {CACHE_FILE} を使用します。")
        df = pd.read_csv(CACHE_FILE)
        
        fix_all_seeds(SEED)
        df_y_test = get_stock_data()
        market_cum = df_y_test.cumsum().mean(axis=1).iloc[-1]
        market_rr = calculate_rr(df_y_test)
        market_stats = (market_cum, market_rr)
    else:
        df, market_stats = process_experiments()
    
    if not df.empty:
        show_average_table(df, market_stats)
        plot_heatmaps(df)