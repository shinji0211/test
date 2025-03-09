import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler  # ここでインポート
from dowhy import CausalModel

# サンプルデータの作成
np.random.seed(42)
n_samples = 100
F = np.random.uniform(10, 100, n_samples)
L = np.random.uniform(1, 10, n_samples)
E = np.random.uniform(1000, 2000, n_samples)
I = np.random.uniform(1e-4, 1e-3, n_samples)

delta = (F * L**3) / (3 * E * I)
data = pd.DataFrame({'F': F, 'L': L, 'E': E, 'I': I, 'delta': delta})

### (A) 線形回帰による寄与度解析 ###
X = data[['F', 'L', 'E', 'I']]
y = data['delta']

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 線形回帰モデルの訓練
reg = LinearRegression()
reg.fit(X_scaled, y)

# 回帰係数を取得
coefficients = pd.DataFrame({
    'Variable': ['F', 'L', 'E', 'I'],
    'Coefficient': reg.coef_
})
print("Linear Regression Coefficients (Standardized):")
print(coefficients)

### (B) 因果推論による寄与度解析 ###
causal_effects = {}
for factor in ['F', 'L', 'E', 'I']:
    model = CausalModel(
        data=data,
        treatment=factor,
        outcome='delta',
        common_causes=[col for col in ['F', 'L', 'E', 'I'] if col != factor]
    )
    identified_estimand = model.identify_effect()
    causal_estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression"  # ここを変更
    )
    causal_effects[factor] = causal_estimate.value

print("Causal Effects (DoWhy):")
for factor, effect in causal_effects.items():
    print(f"{factor}: {effect}")