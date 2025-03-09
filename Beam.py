import numpy as np
import pandas as pd
import dowhy
from dowhy import CausalModel

# パラメータの設定
np.random.seed(42)
n_samples = 100000  # サンプル数

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

print(data.head())


# 因果モデルの作成
model = CausalModel(
    data=data,
    treatment='F',   # 介入変数（荷重）
    outcome='delta',  # 結果変数（たわみ）
    common_causes=['L', 'E', 'I', 'D']  # 共変量（梁の長さ、ヤング率、断面二次モーメント, dummy）
#    common_causes=['L', 'E', 'I']  # 共変量（梁の長さ、ヤング率、断面二次モーメント, dummy）
)

# 因果グラフの表示
model.view_model()

# 因果推論の結果を取得
identified_estimand = model.identify_effect()
causal_estimate = model.estimate_effect(identified_estimand,
                                        method_name="backdoor.linear_regression")

# 結果の表示
print(causal_estimate)
