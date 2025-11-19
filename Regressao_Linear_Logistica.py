# ---------------------------------------------------------------
# PROJETO INDIVIDUAL (PI1) - Machine Learning Supervisionado
# Tema: Predição e Classificação de Aprovados em um Curso Técnico
# Modelos utilizados: Regressão Linear e Regressão Logística
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# ===============================================================
# 1. DESCRIÇÃO DO PROBLEMA
# ===============================================================
"""
O objetivo deste projeto é analisar o desempenho de estudantes em um curso técnico,
utilizando Machine Learning para:

1. Prever a nota final do aluno (Regressão Linear).
2. Classificar se o aluno será aprovado ou reprovado (Regressão Logística).

As variáveis utilizadas incluem:
- Horas de estudo por semana
- Número de faltas
- Participação em aula (0 a 10)
- Nota final (0 a 100)
- Situação final (Aprovado/Reprovado)
"""

# ===============================================================
# 2. PROCESSO DE ETL – CRIAÇÃO & LIMPEZA DO DATASET
# ===============================================================

# Dados fictícios
np.random.seed(42)
n = 200

horas_estudo = np.random.randint(1, 20, n)
faltas = np.random.randint(0, 15, n)
participacao = np.random.randint(1, 10, n)

# Nota final simulada
nota_final = (horas_estudo * 3) + (participacao * 4) - (faltas * 1.5) + np.random.normal(0, 10, n)
nota_final = np.clip(nota_final, 0, 100)

# Aprovado se nota >= 60
aprovado = (nota_final >= 60).astype(int)

df = pd.DataFrame({
    "horas_estudo": horas_estudo,
    "faltas": faltas,
    "participacao": participacao,
    "nota_final": nota_final,
    "aprovado": aprovado
})

print("Primeiras linhas do dataset:")
print(df.head())

# ===============================================================
# 3. ANALISE EXPLORATÓRIA E GRÁFICOS
# ===============================================================

plt.figure(figsize=(6,4))
sns.scatterplot(x=df["horas_estudo"], y=df["nota_final"])
plt.title("Relação entre Horas de Estudo e Nota Final")
plt.xlabel("Horas de Estudo")
plt.ylabel("Nota Final")
plt.show()

plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="Blues")
plt.title("Correlação entre Variáveis")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(df["nota_final"])
plt.title("Distribuição das Notas Finais")
plt.show()

# ===============================================================
# 4. MODELO 1 – REGRESSÃO LINEAR (Predição da Nota Final)
# ===============================================================

X_reg = df[["horas_estudo", "faltas", "participacao"]]
y_reg = df["nota_final"]

X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=0)

modelo_reg = LinearRegression()
modelo_reg.fit(X_train, y_train)

y_pred_reg = modelo_reg.predict(X_test)

print("\n--- RESULTADOS DA REGRESSÃO LINEAR ---")
print("MSE:", mean_squared_error(y_test, y_pred_reg))
print("Coeficientes:", modelo_reg.coef_)

plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred_reg)
plt.xlabel("Nota Real")
plt.ylabel("Nota Prevista")
plt.title("Regressão Linear – Real vs Previsto")
plt.show()

# ===============================================================
# 5. MODELO 2 – REGRESSÃO LOGÍSTICA (Classificar Aprovado/Reprovado)
# ===============================================================

X_clf = df[["horas_estudo", "faltas", "participacao"]]
y_clf = df["aprovado"]

scaler = StandardScaler()
X_clf_scaled = scaler.fit_transform(X_clf)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf_scaled, y_clf, test_size=0.3, random_state=0)

modelo_log = LogisticRegression()
modelo_log.fit(X_train_c, y_train_c)

y_pred_c = modelo_log.predict(X_test_c)

print("\n--- RESULTADOS DA REGRESSÃO LOGÍSTICA ---")
print("Acurácia:", accuracy_score(y_test_c, y_pred_c))
print("\nRelatório de Classificação:")
print(classification_report(y_test_c, y_pred_c))
