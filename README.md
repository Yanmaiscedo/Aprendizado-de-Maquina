# Aprendizado-de-Máquina -- PI1

Este projeto contém vários modelos de Machine Learning desenvolvidos
para o Projeto Individual (PI1), incluindo:

-   Regressão Linear\
-   Regressão Logística\
-   Random Forest\
-   Support Vector Machine (SVM)\
-   KNN\
-   Decision Tree\
-   XGBoost\
-   LightGBM\
-   Rede Neural Artificial (ANN)

------------------------------------------------------------------------

## 🛠️ 1. Preparando o ambiente

### ✔️ 1.1 Instale o Python

Versão recomendada: **Python 3.10 ou 3.11**\
Baixe em: https://www.python.org/downloads/

Durante a instalação, marque:

✔️ **Add Python to PATH**

------------------------------------------------------------------------

## 📂 2. Instalando as bibliotecas necessárias

Abra o terminal do VS Code e execute:

    pip install numpy pandas matplotlib seaborn scikit-learn tensorflow xgboost lightgbm

------------------------------------------------------------------------

## 📦 3. Estrutura de pastas sugerida

    PI1
    │─ Regressao_Linear_Logistica.py
    │─ Random_Forest_SVM.py
    │─ KNN_Árvore_Decisao.py
    │─ XGBoost_LightGBM.py
    │─ ANN.py
    │─ README.md

------------------------------------------------------------------------

## ▶️ 4. Como rodar cada código

Os passos são os mesmos para todos os arquivos:

1.  Abra o **VS Code**
2.  Vá em **File \> Open Folder** e selecione a pasta do projeto
3.  Abra o arquivo `.py` desejado
4.  Abra o terminal do VS Code:\
    **Terminal \> New Terminal**
5.  Execute:

```{=html}
<!-- -->
```
    python nome_do_arquivo.py

------------------------------------------------------------------------

## 🔴 5. Como rodar Regressão Linear + Regressão Logística

**Arquivo:** `Regressao_Linear_Logistica.py`

    python Regressao_Linear_Logistica.py

------------------------------------------------------------------------

## 🟣 6. Como rodar Random Forest + SVM

**Arquivo:** `Random_Forest_SVM.py`

    python Random_Forest_SVM.py

------------------------------------------------------------------------

## 🔵 7. Como rodar KNN + Decision Tree

**Arquivo:** `KNN_Árvore_Decisao.py`

    python KNN_Árvore_Decisao.py

------------------------------------------------------------------------

## 🟠 8. Como rodar XGBoost + LightGBM

**Arquivo:** `XGBoost_LightGBM.py`

    python XGBoost_LightGBM.py

⚠️ Se o LightGBM der erro no Windows, tente:

    pip install lightgbm --install-option=--gpu

------------------------------------------------------------------------

## 🧠 9. Como rodar a Rede Neural Artificial (ANN)

**Arquivo:** `ANN.py`

    python ANN.py

------------------------------------------------------------------------

## 🎨 10. Visualização dos gráficos

Todos os scripts utilizam **matplotlib**, então os gráficos irão abrir
automaticamente ao final da execução.
