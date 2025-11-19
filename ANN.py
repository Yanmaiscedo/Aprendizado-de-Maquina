# -----------------------------------------------
# PI1 – Rede Neural Artificial (ANN)
# Autor: Yan Macedo Teixeira
# -----------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ----------------------------
# 1. GERANDO UM DATASET FICTÍCIO
# ----------------------------

np.random.seed(42)

n = 500

idade = np.random.randint(18, 70, n)
renda = np.random.randint(1500, 15000, n)

# Regra artificial: quanto maior idade + renda, maior chance de comprar
prob_compra = (0.3 * (idade / idade.max())) + (0.7 * (renda / renda.max()))
prob_compra = prob_compra / prob_compra.max()

compra = np.random.binomial(1, prob_compra)

df = pd.DataFrame({
    "idade": idade,
    "renda": renda,
    "comprou": compra
})

print(df.head())

# ----------------------------
# 2. ETL E PREPARAÇÃO
# ----------------------------

# Separando features e target
X = df[["idade", "renda"]]
y = df["comprou"]

# Normalizando
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ----------------------------
# 3. CONSTRUINDO A REDE NEURAL (ANN)
# ----------------------------

model = Sequential()
model.add(Dense(8, activation='relu', input_dim=2))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # saída binária

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

# ----------------------------
# 4. TREINAMENTO
# ----------------------------

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# ----------------------------
# 5. AVALIAÇÃO E RESULTADOS
# ----------------------------

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("\nAcurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

# ----------------------------
# 6. GRÁFICOS DO TREINO
# ----------------------------

plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title("Função de perda durante o treinamento")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title("Acurácia durante o treinamento")
plt.xlabel("Épocas")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# ----------------------------
# 7. MAPA DE CALOR DAS CORRELAÇÕES
# ----------------------------

plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="Blues")
plt.title("Correlação entre variáveis")
plt.show()
