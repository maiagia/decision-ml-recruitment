import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
import requests

# ======= Inicializar session_state com valores padrão =======
st.session_state.setdefault("nivel_profissional_candidato", "Júnior")
st.session_state.setdefault("nivel_ingles_candidato", "Básico")
st.session_state.setdefault("nivel_espanhol_candidato", "Básico")
st.session_state.setdefault("nivel_academico_candidato", "Superior")
st.session_state.setdefault("local_candidato", "São Paulo")
st.session_state.setdefault("cv_texto", "Descreva aqui as experiências e qualificações...")

# ======= Carregar artefatos =======
with open("etl/output/thresholds_dinamicos.json", "r", encoding="utf-8") as f:
    thresholds = json.load(f)

# ======= Carregar vagas =======
with open("etl/data/vagas.json", "r", encoding="utf-8") as f:
    vagas_dict = json.load(f)

# Dicionário de títulos para exibição no selectbox
opcoes_vagas = {
    vaga_id: vagas_dict[vaga_id]["informacoes_basicas"]["titulo_vaga"]
    for vaga_id in vagas_dict
}

# ======= UI Streamlit =======
st.set_page_config(page_title="Decision Match Predictor", layout="wide")
st.title("🔍 Preditor de Match com IA")
st.markdown("Selecione uma vaga abaixo e preencha os dados do candidato para prever o match.")

# ======= Vaga selecionada =======
vaga_id = st.selectbox("Selecione uma vaga:", list(opcoes_vagas.keys()), format_func=lambda vid: opcoes_vagas.get(vid, f"ID {vid}"))
vaga = vagas_dict[vaga_id]

info = vaga["informacoes_basicas"]
perfil = vaga["perfil_vaga"]

# ======= Dados da Vaga =======
with st.expander("📄 Dados da Vaga (apenas leitura)", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.text_input("Nível Profissional da Vaga", perfil.get("nivel profissional", ""), disabled=True)
        st.text_input("Inglês da Vaga", perfil.get("nivel_ingles", ""), disabled=True)
    with col2:
        st.text_input("Espanhol da Vaga", perfil.get("nivel_espanhol", ""), disabled=True)
        st.text_input("Nível Acadêmico da Vaga", perfil.get("nivel_academico", ""), disabled=True)
    with col3:
        st.text_input("Local da Vaga", perfil.get("cidade", "São Paulo"), disabled=True)
        st.text_input("Cliente", info.get("cliente", ""), disabled=True)
        st.text_input("Título da Vaga", info.get("titulo_vaga", ""), disabled=True)
    st.text_area("Requisitos da Vaga", perfil.get("principais_atividades", ""), height=200, disabled=True)

# ======= Dados do Candidato =======
with st.expander("👤 Dados do Candidato", expanded=True):
    col4, col5, col6 = st.columns(3)

    with col4:
        st.selectbox("Nível Profissional do Candidato",
                     ["Júnior", "Pleno", "Sênior", "Outro"],
                     key="nivel_profissional_candidato")
        st.selectbox("Inglês do Candidato",
                     ["Básico", "Intermediário", "Avançado", "Fluente", "Desconhecido"],
                     key="nivel_ingles_candidato")

    with col5:
        st.selectbox("Espanhol do Candidato",
                     ["Básico", "Intermediário", "Avançado", "Fluente", "Desconhecido"],
                     key="nivel_espanhol_candidato")
        st.selectbox("Nível Acadêmico do Candidato",
                     ["Fundamental", "Médio", "Superior", "Pós-graduação", "Mestrado", "Doutorado", "Desconhecido"],
                     key="nivel_academico_candidato")

    with col6:
        st.text_input("Local do Candidato", key="local_candidato")

    st.text_area("Currículo do Candidato", key="cv_texto")

# ======= Predição =======
if st.button("Prever Match"):
    requisitos_vaga = perfil.get("principais_atividades", "").lower()
    cv_texto_lower = st.session_state.cv_texto.lower()

    vJsonRequest = {
        "pDados": [{
            "nivel_profissional_vaga": perfil.get("nivel profissional", ""),
            "nivel_profissional_candidato": st.session_state.nivel_profissional_candidato,
            "nivel_ingles_vaga": perfil.get("nivel_ingles", ""),
            "nivel_espanhol_vaga": perfil.get("nivel_espanhol", ""),
            "nivel_academico_vaga": perfil.get("nivel_academico", ""),
            "nivel_academico_candidato": st.session_state.nivel_academico_candidato,
            "nivel_ingles_candidato": st.session_state.nivel_ingles_candidato,
            "nivel_espanhol_candidato": st.session_state.nivel_espanhol_candidato,
            "local_vaga": perfil.get("cidade", ""),
            "local_candidato": st.session_state.local_candidato,
            "cliente": info.get("cliente", ""),
            "titulo_vaga": info.get("titulo_vaga", ""),
            "requisitos_vaga": requisitos_vaga,
            "cv_texto": cv_texto_lower
        }],
        "pCaminhoModelo": "etl/output/modelo_match_xgb.joblib",
        "pCaminhoPipeline": "etl/output/preprocessador_xgb.joblib",
        "pCaminhoVetorizador": "etl/output/vetorizador_sim_textual.joblib"
    }

    vResposta = requests.post("http://localhost:8000/prever", json=vJsonRequest)

    prob = vResposta.json().get("match", 0.0)
    sim_textual = vResposta.json().get("sim_textual", 0.0)

    grupo = vJsonRequest["pDados"][0]["nivel_profissional_vaga"]
    threshold = thresholds.get(grupo, 0.5)
    pred = int(prob >= threshold)

    # ====== Resultado ======
    if pred == 1:
        st.success("✅ Resultado: Match Potencial")
    else:
        st.error("❌ Resultado: Sem Match Potencial")

    st.markdown(f"**Probabilidade de Match:** `{prob:.2%}`")
    st.markdown(f"**Threshold aplicado:** `{threshold:.2f}`")
    st.markdown(f"**Similaridade Textual:** `{sim_textual:.4f}`")
    st.progress(min(int(prob * 100), 100))
