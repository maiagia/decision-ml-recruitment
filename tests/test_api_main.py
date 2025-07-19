import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from api.main import vApp
import numpy as np

client = TestClient(vApp)

@pytest.fixture
def exemplo_requisicao():
    return {
        "pDados": [{
            "nivel_profissional_vaga": "pleno",
            "nivel_profissional_candidato": "pleno",
            "nivel_ingles_vaga": "intermediario",
            "nivel_ingles_candidato": "intermediario",
            "nivel_espanhol_vaga": "basico",
            "nivel_espanhol_candidato": "basico",
            "local_vaga": "SP",
            "local_candidato": "SP",
            "nivel_academico_vaga": "graduacao",
            "nivel_academico_candidato": "graduacao",
            "requisitos_vaga": "python, dados, sql",
            "cv_texto": "experiência com sql e python"
        }],
        "pCaminhoModelo": "mock_modelo.joblib",
        "pCaminhoPipeline": "mock_pipeline.joblib",
        "pCaminhoVetorizador": "mock_vetorizador.joblib"
    }

@patch("api.main.vML.carregarModeloXGB")
@patch("api.main.vML.carregarPipeline")
@patch("api.main.vML.carregarVetorizadorTextual")
@patch("api.main.vML.calcularSimilaridadeTextual", return_value=[0.88])
@patch("api.main.vML.preverProbabilidades", return_value=np.array([0.73]))
def test_prever_endpoint(mock_prob, mock_sim, mock_vet, mock_pipe, mock_model, exemplo_requisicao):
    resposta = client.post("/prever", json=exemplo_requisicao)
    assert resposta.status_code == 200
    resultado = resposta.json()
    assert "match" in resultado
    assert "sim_textual" in resultado
    assert resultado["match"] == 0.73
    assert resultado["sim_textual"] == 0.88
