import sys
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. パスの設定 (重要) ---
# tradercompany フォルダを読み込めるように、一つ上の階層をパスに追加します
sys.path.append('../')

# もしクラス定義の読み込みでエラーが出る場合は、明示的にimportが必要な場合があります
# from tradercompany.company import Company 
# from tradercompany.trader import Trader

# --- 2. 可視化関数の定義 ---
def visualize_genes(model):
    """
    モデル内の全トレーダーの遺伝子情報を抽出し、統計グラフを描画する関数
    """
    num_factors_list = []
    usage_data = [] # (Input_Stock_ID, Delay_Time) のペアを保存

    print("遺伝子情報を抽出中...")
    # 全トレーダー × 全ターゲット銘柄 の遺伝子を走査
    for i_trader, trader in enumerate(model.traders):
        for i_stock_target in range(model.num_stock):
            try:
                # パラメータ取得
                params = trader.get_params(i_stock_target)
                
                # 1. 項数 (num_factors)
                n_factors = params[0]["num_factor"]
                num_factors_list.append(n_factors)
                
                # 2. 各項の銘柄・遅延情報
                for i_factor in range(n_factors):
                    factor_dict = params[i_factor + 1]
                    
                    if "stock_P" in factor_dict and "delay_P" in factor_dict:
                        usage_data.append({
                            "Used_Stock": int(factor_dict["stock_P"]),
                            "Delay": int(factor_dict["delay_P"]),
                            "Type": "P"
                        })
                    
                    if "stock_Q" in factor_dict and "delay_Q" in factor_dict:
                        usage_data.append({
                            "Used_Stock": int(factor_dict["stock_Q"]),
                            "Delay": int(factor_dict["delay_Q"]),
                            "Type": "Q"
                        })
            except Exception as e:
                continue

    # --- グラフ描画 ---
    plt.figure(figsize=(18, 6))

    # グラフ1: 項数の分布
    plt.subplot(1, 2, 1)
    if num_factors_list:
        max_factors = max(num_factors_list)
        bins = np.arange(0.5, max_factors + 1.5, 1)
        sns.histplot(num_factors_list, bins=bins, kde=False, color="teal")
        plt.title("Distribution of Equation Complexity (Num Factors)", fontsize=14)
        plt.xlabel("Number of Factors", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.xticks(range(1, int(max_factors) + 1))
    else:
        plt.text(0.5, 0.5, "No Data", ha='center')

    # グラフ2: ヒートマップ
    plt.subplot(1, 2, 2)
    if usage_data:
        df_usage = pd.DataFrame(usage_data)
        pivot_table = df_usage.groupby(["Used_Stock", "Delay"]).size().unstack(fill_value=0)
        sns.heatmap(pivot_table, cmap="magma_r", cbar_kws={'label': 'Frequency'})
        plt.title("Feature Usage Heatmap: Stock vs Delay", fontsize=14)
        plt.xlabel("Delay (Days Ago)", fontsize=12)
        plt.ylabel("Input Stock ID", fontsize=12)
    else:
        plt.text(0.5, 0.5, "No Gene Data Found", ha='center')

    plt.tight_layout()
    plt.show()

# --- 3. メイン実行部分 ---
if __name__ == "__main__":
    # ここでファイル名を指定
    model_filename = "model_genetic_algorithm.pkl"
    
    if os.path.exists(model_filename):
        print(f"{model_filename} を読み込んでいます...")
        
        # 【重要】 ここで pickle.load を使ってファイルからオブジェクトに戻します
        with open(model_filename, 'rb') as f:
            trained_model = pickle.load(f)
        
        print("読み込み完了。分析を開始します。")
        
        # 読み込んだオブジェクト(trained_model)を関数に渡す
        visualize_genes(trained_model)
    else:
        print(f"エラー: {model_filename} が見つかりません。同じフォルダに置いてください。")