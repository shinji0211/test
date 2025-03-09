import pandas as pd
import numpy as np

# パラメータの設定
np.random.seed(42)
n_samples = 1000  # サンプル数

# 荷重、梁の長さ、ヤング率、断面二次モーメントのランダム生成
F = np.random.uniform(10, 100, n_samples)  # 荷重 (N)
L = np.random.uniform(1, 10, n_samples)    # 梁の長さ (m)
E = np.random.uniform(200e9, 210e9, n_samples)  # ヤング率 (Pa)
I = np.random.uniform(1e-6, 5e-6, n_samples)  # 断面二次モーメント (m^4)
D = np.linspace(1, 20, n_samples) 

# たわみの計算 (理論式に基づく)
delta = (F * L**3) / (3 * E * I)

# データフレームの作成
data = pd.DataFrame({
    'F': F,  # 荷重
    'L': L,  # 梁の長さ
    'E': E,  # ヤング率
    'I': I,  # 断面二次モーメント
    'D': D,  # 断面二次モーメント
    'delta': delta  # たわみ
})

# CSV ファイルに出力
data.to_csv('output.csv', index=False, encoding='utf-8')

print("データフレームを 'output.csv' として保存しました。")