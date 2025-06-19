
# 🚀 Datathon Pós-Tech: Machine Learning Engineering - Recrutamento com IA

Este repositório contém a solução de **ETL (Extração, Transformação e Carga)** desenvolvida para o desafio do Datathon do curso Pós-Tech FIAP. A proposta é usar IA para otimizar o processo de recrutamento da empresa fictícia **Decision**, que atua na área de bodyshop de TI.

---

## 🧠 Objetivo do Projeto

A empresa enfrenta dificuldades em encontrar e engajar candidatos ideais para vagas de TI. O desafio é estruturar um pipeline que consolide os dados de candidatos, vagas e prospecções, gerando uma base unificada e limpa para alimentar modelos preditivos de "match".

---

## 📂 Base de Dados

A base é composta por três arquivos JSON:

- `vagas.json`: informações de vagas, clientes, localização, requisitos.
- `applicants.json`: dados dos candidatos como nome, formação, idiomas, currículo.
- `prospects.json`: histórico de encaminhamentos por vaga.

---

## ⚙️ Pipeline ETL

### ✅ Extração

Os arquivos JSON são lidos da pasta `data/`:

```python
with open("../data/vagas.json") as f:
    jobs_data = json.load(f)
```

---

### 🔄 Transformação

O processo consolida e padroniza os dados:

- **Normalização de níveis** (idiomas e formação)
- **Limpeza de textos** (currículo e requisitos)
- **Criação do campo-alvo `match`**: `1` se o candidato foi contratado, `0` caso contrário.
- **Tratamento de campos ausentes com valores padrão**

Exemplo de limpeza de requisitos:

```python
requisitos = f"{tecnicos} {atividades}"
requisitos = limpar_texto(requisitos)
```

---

### 💾 Carga

O `DataFrame` final é salvo em:

```python
../output/dataset_unificado.csv
```

E pode ser visualizado com:

```python
df.head(100)
```

---

## 📁 Estrutura do Projeto

```
├── etl/
│   ├── data/               # Arquivos JSON de entrada
│   ├── output/             # Arquivo CSV final consolidado
│   ├── notebooks/          # Análises e testes manuais
│   ├── etl_pipeline.py     # Lógica principal do ETL
│   └── requirements.txt    # Bibliotecas usadas
```

---

## ✅ Como Executar

1. Clone este repositório:
```bash
git clone https://github.com/seu-usuario/api-etl-pipeline.git
cd api-etl-pipeline/etl
```

2. Ative o ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate.bat  # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Execute o script:
```bash
python etl_pipeline.py
```

O arquivo `dataset_unificado.csv` será gerado na pasta `output/`.

---
