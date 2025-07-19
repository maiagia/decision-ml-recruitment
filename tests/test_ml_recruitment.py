
import pytest
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from api_interna.src.ml_recruitment import ML_Recruitment

@pytest.fixture
def ml():
    return ML_Recruitment()

def test_base_set_get():
    df = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
    ml = ML_Recruitment()
    ml.baseDeDados = df
    assert isinstance(ml.baseDeDados, pd.DataFrame)
    assert ml.baseDeDados.equals(df)

def test_padronizar_colunas():
    df = pd.DataFrame({'Nome Completo': ['João', 'Maria'], 'Idade': [25, 30]})
    ml = ML_Recruitment()
    ml.baseDeDados = df
    ml.padronizarBase()
    assert 'nome_completo' in ml.baseDeDados.columns
    assert 'idade' in ml.baseDeDados.columns

def test_vetorizador_textual():
    df = pd.DataFrame({'col1': ['ola'], 'col2': ['ola']})
    ml = ML_Recruitment()
    ml.baseDeDados = df
    ml.treinarVetorizadorTextual(['col1', 'col2'])
    assert isinstance(ml.vetorizadorTextual, TfidfVectorizer)

def test_similaridade_textual():
    df = pd.DataFrame({'col1': ['bom dia'], 'col2': ['bom dia']})
    ml = ML_Recruitment()
    ml.baseDeDados = df
    ml.treinarVetorizadorTextual(['col1', 'col2'])
    sim = ml.calcularSimilaridadeTextual(['col1', 'col2'])
    assert isinstance(sim, np.ndarray)
    assert len(sim) == len(df)

def test_separar_feature_target():
    df = pd.DataFrame({
        'texto': ['a', 'b', 'c'],
        'numero': [1, 2, 3],
        'classe': [0, 1, 0]
    })
    ml = ML_Recruitment()
    ml.baseDeDados = df
    ml.separarFeatureTarget('classe')
    assert 'classe' not in ml.features.columns
    assert ml.target.equals(df['classe'])

def test_pipeline_e_execucao():
    df = pd.DataFrame({
        'num': [1, 2, 3, 4],
        'cat': ['a', 'b', 'a', 'b'],
        'txt1': ['ola mundo', 'bom dia', 'boa noite', 'bom dia'],
        'txt2': ['oi mundo', 'boa tarde', 'bom dia', 'boa tarde'],
        'target': [0, 1, 0, 1]
    })
    ml = ML_Recruitment()
    ml.baseDeDados = df
    ml.separarFeatureTarget('target')
    ml.criarPipeline(['num'], ['cat'], ['txt1', 'txt2'], 10)
    ml.separarTreinoTeste(0.5)
    ml.executarPipeline('Treino_Teste')
    assert hasattr(ml, 'features_Treino_Transformadas')
    assert hasattr(ml, 'features_Teste_Transformadas')


def test_modelo_xgb_e_previsao():
    df = pd.DataFrame({
        'num': [1, 2, 3, 4],
        'cat': ['a', 'b', 'a', 'b'],
        'txt1': ['ola mundo', 'bom dia', 'boa noite', 'bom dia'],
        'txt2': ['oi mundo', 'boa tarde', 'bom dia', 'boa tarde'],
        'target': [0, 1, 0, 1]
    })
    ml = ML_Recruitment()
    ml.baseDeDados = df
    ml.separarFeatureTarget('target')
    ml.criarPipeline(['num'], ['cat'], ['txt1', 'txt2'], 10)
    ml.separarTreinoTeste(0.5)
    ml.executarPipeline('Treino_Teste')
    ml.criarModeloXGB(pTreinarModelo=True)
    relatorio = ml.avaliarModeloXGB()
    assert isinstance(relatorio, str)
    previsao = ml.prever(ml.features_Teste)
    assert isinstance(previsao, np.ndarray)
    probas = ml.preverProbabilidades(ml.features_Teste)
    assert isinstance(probas, np.ndarray)
    assert len(previsao) == len(probas)
