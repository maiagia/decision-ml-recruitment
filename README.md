
# 🚀 Datathon Pós-Tech: Machine Learning Engineering - Recrutamento com IA

👨‍💻 Equipe

Kleryton de Souza, Lucas Paim, Maiara Giavoni, Rafael Tafelli

# Decision Match Predictor - IA para Recrutamento Inteligente

Este projeto utiliza técnicas de Machine Learning para prever o "match" entre candidatos e vagas reais da empresa **Decision**, especializada em recrutamento no setor de TI.

---

## 📌 Visão Geral

- O objetivo é automatizar parte do processo seletivo, ajudando hunters a identificar candidatos com maior potencial de contratação.
- O modelo é treinado com dados históricos (candidatos, vagas e prospects) e considera múltiplas features estruturadas e textuais.
- O app em **Streamlit** permite testar novos candidatos de forma interativa.

---

## 🧱 Estrutura do Projeto

```
├── etl_dataset_match.ipynb         # Notebook para construção do dataset consolidado e balanceado
├── modelo.ipynb                    # Notebook com pipeline de treinamento, ajuste de threshold e salvamento dos artefatos
├── app_streamlit.py                # Interface interativa para inferência do modelo
├── output/
│   ├── modelo_match_xgb.joblib             # Modelo treinado (XGBoost)
│   ├── preprocessador_xgb.joblib           # Pipeline de pré-processamento (ColumnTransformer)
│   ├── vetorizador_sim_textual.joblib      # Vetorizar TF-IDF treinado para similaridade textual
│   ├── dataset_unificado.csv               # Dataset original consolidado
│   ├── dataset_unificado_balanceado.csv    # Dataset balanceado (50/50 match e não match)
│   └── exemplos_para_teste_app.json        # Amostras reais para validação no app
└── data/
    ├── vagas.json
    ├── applicants.json
    └── prospects.json
```

---

## 🚀 Como Executar

### 1. Instalar Dependências

Crie um ambiente virtual (opcional) e instale os requisitos:

```bash
pip install -r requirements.txt
```

### 2. Executar o App

```bash
streamlit run app_streamlit.py
```

O app será iniciado em `http://localhost:8501`.

---

## 📊 O que o modelo considera?

- Nível profissional, inglês, espanhol, acadêmico e local (vaga vs candidato)
- Similaridade textual entre os requisitos da vaga e o currículo
- Feature de similaridade ponderada (peso 0.3) para evitar overfitting em texto

---

## 🔍 Testes com Casos Reais

- Os **10 cargos mais populares** foram selecionados com base nas candidaturas.
- Para cada vaga, foi salvo:
  - 1 exemplo real de **match**
  - 1 exemplo real de **não-match**
- Esses dados estão em `output/exemplos_para_teste_app.json`.

---