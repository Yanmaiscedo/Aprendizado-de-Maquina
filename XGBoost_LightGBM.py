# ---------------------------------------------------------------
# PROJETO 4 - Previsão de Atraso em Pagamento (Default)
# Modelos utilizados: XGBoost + LightGBM
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ===============================================================
# 1. DESCRIÇÃO DO PROBLEMA
# ===============================================================
"""
Objetivo:
Prever se um cliente irá atrasar um pagamento (default = 1)
com base em características de perfil financeiro.

Variáveis:
- renda
- idade
- score_credito
- dividas_totais
- meses_atraso_ultimo_ano
- default (0 = não atrasou, 1 = atrasou)
"""

# ===============================================================
# 2. CRIAÇÃO DO DATASET (fictício)
# ===============================================================

np.random.seed(42)
n = 500

renda = np.random.normal(4000, 1500, n).clip(800, 15000)
idade = np.random.randint(18, 70, n)
score_credito = np.random.randint(300, 1000, n)
dividas_totais = np.random.normal(5000, 3000, n).clip(0, 50000)
meses_atraso = np.random.randint(0, 12, n)

# regra fictícia pro default
default = (
    (renda < 2500) |
    (score_credito < 500) |
    (dividas_totais > 20000) |
    (meses_atraso > 4)
).astype(int)

df = pd.DataFrame({
    "renda": renda,
    "idade": idade,
    "score_credito": score_credito,
    "dividas_totais": dividas_totais,
    "meses_atraso": meses_atraso,
    "default": default
})

print(df.head())

# ===============================================================
# 3. ANÁLISE EXPLORATÓRIA
# ===============================================================

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="default")
plt.title("Distribuição do Default")
plt.show()

plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="Oranges")
plt.title("Correlação entre Variáveis")
plt.show()

# ===============================================================
# 4. PREPARAÇÃO DOS DADOS
# ===============================================================

X = df.drop("default", axis=1)
y = df["default"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=0
)

# ===============================================================
# 5. MODELO 1 – XGBOOST
# ===============================================================

modelo_xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    eval_metric="logloss"
)

modelo_xgb.fit(X_train, y_train)
y_pred_xgb = modelo_xgb.predict(X_test)

print("\n--- XGBoost ---")
print("Acurácia:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))

# ===============================================================
# 6. MODELO 2 – LIGHTGBM
# ===============================================================

modelo_lgb = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=-1
)

modelo_lgb.fit(X_train, y_train)
y_pred_lgb = modelo_lgb.predict(X_test)

print("\n--- LightGBM ---")
print("Acurácia:", accuracy_score(y_test, y_pred_lgb))
print(classification_report(y_test, y_pred_lgb))

# ===============================================================
# 7. COMPARAÇÃO GRÁFICA
# ===============================================================

plt.figure(figsize=(6,4))
plt.bar(
    ["XGBoost", "LightGBM"],
    [accuracy_score(y_test, y_pred_xgb), accuracy_score(y_test, y_pred_lgb)]
)
plt.ylabel("Acurácia")
plt.title("Comparação de Desempenho - XGBoost vs LightGBM")
plt.show()
