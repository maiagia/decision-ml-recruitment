# from ml_recruitment import ML_Recruitment
from api_interna.src.ml_recruitment import ML_Recruitment
from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
import pandas as pd

vApp = FastAPI()
vML = ML_Recruitment()

class PreverRequest(BaseModel):
    """
    Modelo de requisição para prever o match entre candidato e vaga.
    """
    pDados: List[dict]
    pCaminhoModelo: str
    pCaminhoPipeline: str
    pCaminhoVetorizador: str

@vApp.post("/prever")
def preverMatch(pRequisicao: PreverRequest) -> dict:
    """
    Endpoint para prever o match entre candidato e vaga.
    """
    # Carregar modelos
    vML.carregarModeloXGB(pRequisicao.pCaminhoModelo)
    vML.carregarPipeline(pRequisicao.pCaminhoPipeline)
    vML.carregarVetorizadorTextual(pRequisicao.pCaminhoVetorizador)

    # Carregar dados
    vDados = pd.DataFrame(pRequisicao.pDados)
    vML.baseDeDados = vDados

    # Padronizar base de dados
    vML.padronizarBase(pListaDeColunasParaFillNA=['requisitos_vaga', 'cv_texto'], pPreencherNulosCom='vazio', pPadronizarColunasTexto=False)
    
    # Features auxiliares
    vML.baseDeDados["match_nivel"] = (vML.baseDeDados["nivel_profissional_vaga"] == vML.baseDeDados["nivel_profissional_candidato"]).astype(int)
    vML.baseDeDados["match_profissional"] = (vML.baseDeDados["nivel_profissional_vaga"] == vML.baseDeDados["nivel_profissional_candidato"]).astype(int)
    vML.baseDeDados["match_ingles"] = (vML.baseDeDados["nivel_ingles_vaga"] == vML.baseDeDados["nivel_ingles_candidato"]).astype(int)
    vML.baseDeDados["match_espanhol"] = (vML.baseDeDados["nivel_espanhol_vaga"] == vML.baseDeDados["nivel_espanhol_candidato"]).astype(int)
    vML.baseDeDados["match_local"] = (vML.baseDeDados["local_vaga"] == vML.baseDeDados["local_candidato"]).astype(int)
    vML.baseDeDados["match_academico"] = (vML.baseDeDados["nivel_academico_vaga"] == vML.baseDeDados["nivel_academico_candidato"]).astype(int)

    # Calcular similaridade textual
    vML.baseDeDados['sim_textual'] = vML.calcularSimilaridadeTextual(pListaColunas=['requisitos_vaga', 'cv_texto'])

    vResultado = vML.preverProbabilidades(pFeatures=vDados)
    print({
        "match": vResultado.tolist()[0],
        "sim_textual": vML.baseDeDados['sim_textual'].values[0]
            })
    return {
        "match": vResultado.tolist()[0],
        "sim_textual": vML.baseDeDados['sim_textual'].values[0]
            }