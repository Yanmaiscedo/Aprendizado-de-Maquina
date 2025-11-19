# ---------------------------------------------------------------
# PROJETO 2 - Classificação de Risco de Crédito
# Modelos utilizados: Random Forest + SVM
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# ===============================================================
# 1. DESCRIÇÃO DO PROBLEMA
# ===============================================================
"""
Objetivo:
Criar um modelo de Machine Learning para classificar clientes como
"Baixo risco" ou "Alto risco" de crédito.

Variáveis:
- renda_mensal
- idade
- score_serasa
- tempo_emprego (anos)
- dividas (valor total)
- risco (0 = baixo risco, 1 = alto risco)
"""

# ===============================================================
# 2. CRIAÇÃO & LIMPEZA DO DATASET (dados fictícios)
# ===============================================================

np.random.seed(42)
n = 400

renda = np.random.normal(3500, 1200, n).clip(800, 15000)
idade = np.random.randint(18, 70, n)
score = np.random.randint(200, 1000, n)
tempo_emprego = np.random.randint(0, 30, n)
dividas = np.random.normal(3000, 2500, n).clip(0, 40000)

# Regra artificial para risco
risco = ((renda < 2500) | (score < 500) | (dividas > 15000)).astype(int)

df = pd.DataFrame({
    "renda_mensal": renda,
    "idade": idade,
    "score_serasa": score,
    "tempo_emprego": tempo_emprego,
    "dividas": dividas,
    "risco": risco
})

print("Primeiras linhas do dataset:")
print(df.head())

# ===============================================================
# 3. ANÁLISE EXPLORATÓRIA
# ===============================================================

plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="risco", y="renda_mensal")
plt.title("Distribuição da Renda por Classe de Risco")
plt.show()

plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="Blues")
plt.title("Correlação entre Variáveis")
plt.show()

# ===============================================================
# 4. PREPARAÇÃO DOS DADOS
# ===============================================================

X = df.drop("risco", axis=1)
y = df["risco"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=0
)

# ===============================================================
# 5. MODELO 1 – RANDOM FOREST
# ===============================================================

modelo_rf = RandomForestClassifier(n_estimators=150, random_state=0)
modelo_rf.fit(X_train, y_train)
y_pred_rf = modelo_rf.predict(X_test)

print("\n--- RANDOM FOREST ---")
print("Acurácia:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# ===============================================================
# 6. MODELO 2 – SUPPORT VECTOR MACHINE (SVM)
# ===============================================================

modelo_svm = SVC(kernel='rbf', C=2, gamma='scale')
modelo_svm.fit(X_train, y_train)
y_pred_svm = modelo_svm.predict(X_test)

print("\n--- SVM ---")
print("Acurácia:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# Gráfico comparando acurácia
plt.figure(figsize=(6,4))
plt.bar(["Random Forest", "SVM"], [accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_svm)])
plt.title("Comparação de Acurácia")
plt.ylabel("Acurácia")
plt.show()
