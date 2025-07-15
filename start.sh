#!/bin/bash

# Inicia FastAPI em background
uvicorn api.main:vApp --host 0.0.0.0 --port 8000 &

# Inicia Streamlit
streamlit run etl/app_streamlit.py --server.port=8501 --server.address=0.0.0.0 &

# Mantém o container vivo esperando os processos
wait
