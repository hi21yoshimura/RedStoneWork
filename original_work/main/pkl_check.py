import pickle
import os
import sys

sys.path.append(r'C:\Users\Y.Yoshimura\.vscode\GitHubRepository\RedStoneWork-1\original_work')

# パスは環境に合わせて修正してください
base_path = r"C:\Users\Y.Yoshimura\.vscode\GitHubRepository\RedStoneWork-1\original_work\model_genetic_algorithm.pkl"
base_path = r"C:\Users\Y.Yoshimura\.vscode\GitHubRepository\RedStoneWork-1\original_work\main"
target_file = os.path.join(base_path, "model_genetic_algorithm.pkl") # 1.pklを確認

if os.path.exists(target_file):
    with open(target_file, "rb") as f:
        data = pickle.load(f)
    print(f"データの型: {type(data)}")
    if isinstance(data, dict):
        print(f"キー一覧: {data.keys()}")
    else:
        print("辞書型ではありません。中身はCompanyオブジェクト等の可能性があります。")
else:
    print(f"ファイルが見つかりません: {target_file}")