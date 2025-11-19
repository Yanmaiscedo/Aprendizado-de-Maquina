# Aprendizado-de-Maquina
Este projeto contém vários modelos de Machine Learning conforme solicitado no PI1:
- KNN
- Decision Tree
- XGBoost
- LightGBM
- Rede Neural Artificial (ANN)

---
🛠️ 1. Preparando o ambiente

✔️ 1.1 Instale o Python

Versão recomendada: Python 3.10 ou 3.11

Baixar em: https://www.python.org/downloads/

Durante a instalação, marque a opção:

✔️ Add Python to PATH.

---

📂 2. Instalando as bibliotecas necessárias

Abra o terminal do VS Code e execute:

- pip install numpy pandas matplotlib seaborn scikit-learn tensorflow xgboost lightgbm

---

📦 3. Estrutura de pastas sugerida

PI1

─ knn_decision_tree.py

─ xgboost_lightgbm.py

─ ann.py

─ README.md

---

▶️ 4. Como rodar cada código

Os passos são os mesmos para todos:

Abra o VS Code

Vá em File > Open Folder e selecione a pasta do projeto

Clique no arquivo .py que você quer rodar

No VS Code, abra um terminal:

Terminal > New Terminal

Execute:

- python nome_do_arquivo.py

---

🔵 5. Como rodar o arquivo KNN + Decision Tree

Nome do arquivo: KNN_Árvore_Decisao.py

Terminal:

- python KNN_Árvore_Decisao.py


Esse script inclui:

✔️ ETL

✔️ Treinamento KNN

✔️ Treinamento Decision Tree

✔️ Gráficos

✔️ Comparação de desempenho

---

🟠 6. Como rodar o arquivo XGBoost + LightGBM

Nome do arquivo: XGBoost_LightGBM.py

Terminal:

- python XGBoost_LightGBM.py


Esse script inclui:

✔️ Treino com XGBoost

✔️ Treino com LightGBM

✔️ ETL + limpeza

✔️ Gráficos

✔️ Comparação dos modelos

⚠️ Observação importante no Windows:

Se o LightGBM der erro na instalação, use:

pip install lightgbm --install-option=--gpu

---

🧠 7. Como rodar a Rede Neural Artificial (ANN)

Nome do arquivo: ANN.py

Terminal:

python ANN.py


Esse script inclui:

✔️ Geração de dados

✔️ Normalização

✔️ Treinamento da rede neural

✔️ Gráficos de loss e accuracy

✔️ Relatório de classificação


[...]
