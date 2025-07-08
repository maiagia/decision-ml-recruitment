import streamlit as st
import pandas as pd
import joblib

# 💾 Carrega o modelo
modelo = joblib.load("output/modelo_match_rf.joblib")

# 🌟 Configuração do app
st.set_page_config(page_title="Decision Match Predictor", layout="wide")
st.title("🚀 Decision - Preditor de Match com IA")
st.markdown("Preencha os dados abaixo para prever se o candidato tem potencial match.")

# 📝 Inputs
nivel_vaga = st.selectbox("Nível da Vaga", ["Júnior", "Pleno", "Sênior", "Outro"])
nivel_ingles_vaga = st.selectbox("Inglês da Vaga", ["Básico", "Intermediário", "Avançado", "Fluente", "Outro"])
nivel_espanhol_vaga = st.selectbox("Espanhol da Vaga", ["Básico", "Intermediário", "Avançado", "Fluente", "Outro"])
nivel_academico = st.selectbox("Nível Acadêmico", ["Fundamental", "Médio", "Superior", "Pós-graduação", "Mestrado", "Doutorado", "Desconhecido"])
nivel_ingles = st.selectbox("Inglês do Candidato", ["Básico", "Intermediário", "Avançado", "Fluente", "Desconhecido"])
cliente = st.text_input("Cliente", "Morris, Moran and Dodson")
local_vaga = st.text_input("Local da Vaga", "São Paulo")
requisitos_vaga = st.text_area("Requisitos da Vaga", "Experiência em gestão de projetos, conhecimento em AWS")
cv_texto = st.text_area("Currículo do Candidato", "Experiência com projetos ágeis, certificação PMP, fluente em inglês")

# 🚀 Botão de predição
if st.button("🔍 Prever Match"):
    dados = pd.DataFrame([{
        "nivel_vaga": nivel_vaga,
        "nivel_ingles_vaga": nivel_ingles_vaga,
        "nivel_espanhol_vaga": nivel_espanhol_vaga,
        "nivel_academico": nivel_academico,
        "nivel_ingles": nivel_ingles,
        "cliente": cliente,
        "local_vaga": local_vaga,
        "requisitos_vaga": requisitos_vaga,
        "cv_texto": cv_texto
    }])

    # Predição
    prob = modelo.predict_proba(dados)[0][1]
    threshold = 0.3
    pred = 1 if prob >= threshold else 0

    # Exibe resultado com cor
    if pred == 1:
        st.success(f"🎯 Resultado: ✅ Match Potencial")
    else:
        st.error(f"🎯 Resultado: ❌ Sem Match Potencial")

    st.markdown(f"**Probabilidade de Match:** {prob:.2%}")

    # Progress bar visual
    st.progress(min(int(prob * 100), 100))

