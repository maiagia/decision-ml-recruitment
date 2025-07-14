FROM python:3.11

# Define diretório de trabalho
WORKDIR /app-streamlit

# Copia todos os arquivos para dentro da imagem
COPY . /app-streamlit

# Atualiza o pip
RUN pip install --upgrade pip

WORKDIR /app-streamlit/api_interna

RUN pip install -r requirements.txt

RUN pip install -e .

WORKDIR /app-streamlit/etl

# Instala as dependências adicionais da aplicação
RUN pip install -r requirements.txt

# Comando para iniciar o app Streamlit
CMD ["streamlit", "run", "app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
