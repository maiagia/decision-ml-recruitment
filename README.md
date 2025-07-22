# 🚀 Datathon Pós-Tech: Machine Learning Engineering - Recrutamento com IA

## 👨‍💻 Equipe
- Kleryton de Souza  
- Lucas Paim  
- Maiara Giavoni  
- Rafael Tafelli  

---

## 🎯 Projeto: *Decision Match Predictor*  
**IA para Recrutamento Inteligente no Setor de TI**

Este projeto propõe uma solução prática e escalável para otimizar o processo de recrutamento da empresa fictícia *Decision*, especializada em bodyshop de tecnologia. Utilizamos técnicas avançadas de Machine Learning para prever o *match* entre candidatos e vagas, com base em fatores técnicos, culturais e motivacionais.

---

## 📌 Visão Geral

✅ Automatização parcial do processo seletivo, auxiliando hunters na triagem de candidatos.  
✅ Uso de dados reais anonimizados de vagas, candidatos e prospects.  
✅ Pipeline completo com **XGBoost**, **TF-IDF**, **normalização** e **engenharia de features**.  
✅ Interface amigável com **Streamlit** para testes manuais.  
✅ API REST com **FastAPI** para integração em produção.  
✅ Empacotamento com **Docker** e logging básico.  
✅ Testes unitários com **pytest**.  

---

## 🧱 Estrutura do Projeto

```bash
├── etl/
│   ├── etl_dataset_match.ipynb              # Geração do dataset unificado
│   ├── output/
│   │   ├── modelo_match_xgb.joblib          # Modelo XGBoost treinado
│   │   ├── preprocessador_xgb.joblib        # ColumnTransformer completo
│   │   ├── vetorizador_sim_textual.joblib   # Vetorizador TF-IDF
│   │   ├── dataset_unificado.csv
│   │   ├── dataset_unificado_balanceado.csv
│   │   └── exemplos_para_teste_app.json     # Casos reais de match e não-match
│   ├── data/
│   │   ├── vagas.json
│   │   ├── applicants.json
│   │   └── prospects.json
│   └── log_inferencias.log
│
├── api/
│   └── main.py                              # API principal com FastAPI
│
├── api_interna/
│   └── ml_recruitment.py, utils.py, etc.    # Módulos de lógica do modelo
│
├── app_streamlit.py                         # Interface Streamlit para validação manual
├── docker.txt                               # Comandos Docker
├── Dockerfile                               # Imagem Docker para API + Streamlit
├── requirements.txt                         # Dependências do projeto
├── start.sh                                 # Script para iniciar aplicação
├── tests/
│   └── test_api_main.py, test_ml_recruitment.py  # Testes unitários
└── README.md
```

---

## 🧪 Como Executar o Projeto

### 1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/decision-ml-recruitment.git
cd decision-ml-recruitment
```

### 2. Crie um ambiente virtual (opcional)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Execute via Streamlit
```bash
streamlit run app_streamlit.py
```
Acesse em: [http://localhost:8501](http://localhost:8501)

---

## 🌐 Executando a API

### 1. Inicie a API FastAPI localmente
```bash
uvicorn api.main:app --reload --port 8000
```

### 2. Teste o endpoint `/prever` com ferramentas como Postman ou via Streamlit

---

## 🐳 Executando com Docker

### 1. Build da imagem
```bash
docker build -t decision-ml-api .
```

### 2. Run do container
```bash
docker run -p 8000:8000 -p 8501:8501 decision-ml-api
```

---

## 📊 O que o modelo considera?

- Nível profissional do candidato vs exigência da vaga  
- Idiomas (inglês e espanhol)  
- Localização e formação acadêmica  
- Similaridade textual entre requisitos da vaga e currículo (com TF-IDF)  
- Feature de similaridade com peso de 0.3 na predição final  
- Thresholds dinâmicos ajustados por nível de vaga com recall mínimo de 0.6  

---

## 🔍 Testes com Casos Reais

Selecionamos os 10 cargos mais comuns e para cada vaga:

- ✅ 1 exemplo real de *match*
- ❌ 1 exemplo real de *não-match*

Esses exemplos estão salvos em:
```
output/exemplos_para_teste_app.json
```

---

## 🧪 Testes Unitários

Incluímos testes para:

- Validação dos endpoints da API (`/prever`)  
- Testes de integridade da pipeline de ML  

Executar com:
```bash
pytest tests/
```

---

## 📹 Demonstração em Vídeo

👉 Apresentação do projeto:  
📺 *[Inserir link do YouTube ou Google Drive]*

---

## 📈 Monitoramento

- As inferências realizadas via API são logadas em:
```
etl/log_inferencias.log
```

- Futuramente, pode-se integrar com Prometheus, Grafana ou ELK Stack para observabilidade avançada.

---

## 🧠 Técnicas e Tecnologias Utilizadas

- Machine Learning com **XGBoost**
- Vetorização textual com **TF-IDF**
- Engenharia de atributos e tratamento de dados com **Pandas + ColumnTransformer**
- APIs com **FastAPI**
- Frontend de inferência com **Streamlit**
- Empacotamento com **Docker**
- **Logging**, **testes unitários** e boas práticas de engenharia

---

## 🏁 Conclusão

A solução *Decision Match Predictor* entrega valor imediato para empresas de recrutamento tech, automatizando a triagem e priorização de candidatos. Ao integrar múltiplas fontes de dados e aplicar um modelo robusto, aumentamos a assertividade do processo seletivo, poupando tempo dos recrutadores e melhorando a experiência dos candidatos.

---
