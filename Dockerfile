FROM python:3.11

# Define diretório de trabalho
WORKDIR /app-streamlit

# Copia tudo para dentro da imagem
COPY . /app-streamlit

# Atualiza o pip
RUN pip install --upgrade pip

# Instala dependências da FastAPI
WORKDIR /app-streamlit/api_interna
RUN pip install -r requirements.txt
RUN pip install -e .

# Instala dependências do Streamlit
WORKDIR /app-streamlit/etl
RUN pip install -r requirements.txt

# Volta para a raiz do projeto
WORKDIR /app-streamlit

# Habilita o script de start (já foi copiado com o COPY . /app-streamlit)
RUN chmod +x /app-streamlit/start.sh

# Expõe as portas usadas pelos dois apps
EXPOSE 8000 8501

# Executa os dois apps
CMD ["./start.sh"]
