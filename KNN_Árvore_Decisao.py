# ---------------------------------------------------------------
# PROJETO 3 - Predição de Churn (Cancelamento de Clientes)
# Modelos utilizados: KNN + Decision Tree
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

# ===============================================================
# 1. DESCRIÇÃO DO PROBLEMA
# ===============================================================
"""
Objetivo:
Criar um modelo de Machine Learning para prever o churn (cancelamento)
de clientes de uma empresa de telefonia.

Variáveis do dataset (fictício):
- idade
- mensalidade
- tempo_cliente (meses)
- suporte_tecnico (0=ruim, 1=ok, 2=ótimo)
- limite_plano (GB de internet)
- churn (0 = não cancelou, 1 = cancelou)
"""

# ===============================================================
# 2. CRIAÇÃO DO DATASET
# ===============================================================

np.random.seed(42)
n = 350

idade = np.random.randint(18, 75, n)
mensalidade = np.random.uniform(50, 300, n)
tempo_cliente = np.random.randint(1, 60, n)
suporte_tecnico = np.random.randint(0, 3, n)
limite_plano = np.random.randint(5, 50, n)

# Regras artificiais para churn
churn = (
    (mensalidade > 220) |
    (suporte_tecnico == 0) |
    (tempo_cliente < 10)
).astype(int)

df = pd.DataFrame({
    "idade": idade,
    "mensalidade": mensalidade,
    "tempo_cliente": tempo_cliente,
    "suporte_tecnico": suporte_tecnico,
    "limite_plano": limite_plano,
    "churn": churn
})

print("Primeiras linhas do dataset:")
print(df.head())

# ===============================================================
# 3. ANÁLISE EXPLORATÓRIA
# ===============================================================

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="churn")
plt.title("Distribuição do Churn (Cancelamento)")
plt.show()

plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="Greens")
plt.title("Correlação entre Variáveis")
plt.show()

# ===============================================================
# 4. PREPARAÇÃO DOS DADOS
# ===============================================================

X = df.drop("churn", axis=1)
y = df["churn"]

# KNN exige padronização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=0
)

# ===============================================================
# 5. MODELO 1 – KNN
# ===============================================================

modelo_knn = KNeighborsClassifier(n_neighbors=7)
modelo_knn.fit(X_train, y_train)
y_pred_knn = modelo_knn.predict(X_test)

print("\n--- KNN ---")
print("Acurácia:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# ===============================================================
# 6. MODELO 2 – ÁRVORE DE DECISÃO
# ===============================================================

modelo_tree = DecisionTreeClassifier(max_depth=5, random_state=0)
modelo_tree.fit(X_train, y_train)
y_pred_tree = modelo_tree.predict(X_test)

print("\n--- DECISION TREE ---")
print("Acurácia:", accuracy_score(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))

# Visualização da árvore
plt.figure(figsize=(14,8))
plot_tree(
    modelo_tree,
    feature_names=X.columns,
    class_names=["Não Cancela", "Cancela"],
    filled=True,
    rounded=True
)
plt.show()

# ===============================================================
# 7. COMPARAÇÃO FINAL (Gráfico)
# ===============================================================

plt.figure(figsize=(6,4))
plt.bar(
    ["KNN", "Decision Tree"],
    [accuracy_score(y_test, y_pred_knn), accuracy_score(y_test, y_pred_tree)]
)
plt.ylabel("Acurácia")
plt.title("Comparação de Desempenho - KNN vs Decision Tree")
plt.show()
