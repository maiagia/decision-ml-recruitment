import pandas as pd
import numpy as np
from unicodedata import normalize, combining
from re import sub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib

class ML_Recruitment:

    __vCaminhoBase: str = ''
    __vBase: pd.DataFrame = None
    __vVetorizador: TfidfVectorizer = None
    __vFeatures: pd.DataFrame = None
    __vTarget: pd.Series = None
    __vPipeline: ColumnTransformer = None
    __vFeatures_Treino: pd.DataFrame = None
    __vTarget_Treino: pd.Series = None
    __vFeatures_Teste: pd.DataFrame = None
    __vTarget_Teste: pd.Series = None
    __vModeloXGB: XGBClassifier = None
    __vFeatures_Treino_Transformadas: np.array = None
    __vFeatures_Teste_Transformadas: np.array = None

    def __init__(self, pCaminhoBase: str = '') -> None:
        self.caminhoBase = pCaminhoBase

    @property
    def caminhoBase(self) -> str:
        return self.__vCaminhoBase
    
    @caminhoBase.setter
    def caminhoBase(self, pCaminhoBase: str):
        self.__vCaminhoBase = pCaminhoBase

    @property
    def baseDeDados(self) -> pd.DataFrame:
        """
        Retorna a base de dados carregada.
        """
        if self.__vBase is None:
            raise ValueError("Base de dados não foi carregada. Use o método 'carregarBase' primeiro.")
        
        return self.__vBase
    
    @baseDeDados.setter
    def baseDeDados(self, pBase: pd.DataFrame):
        """
        Define a base de dados.
        """
        if not isinstance(pBase, pd.DataFrame):
            raise TypeError("A base de dados deve ser um DataFrame do pandas.")
        
        self.__vBase = pBase

    def carregarBase(self) -> None:
        """
        Carrega a base de dados do caminho especificado.
        """
        if self.caminhoBase == '':
            raise ValueError("Caminho da base de dados não foi especificado.")
        
        self.__vBase = pd.read_csv(self.caminhoBase)

    def __str__(self):
        return (
            f'Caminho da Base: {self.caminhoBase}\n'
            f'Número de Linhas: {self.baseDeDados.shape[0]}\n'
            f'Número de Colunas: {self.baseDeDados.shape[1]}\n'
            f'Tamanho das features: {self.features.shape if self.features is not None else "Não definido"}\n'
            f'Tamanho do target: {self.target.shape if self.target is not None else "Não definido"}\n'
            f'Número de Features de Treino: {self.features_Treino.shape if self.features_Treino is not None else "Não definido"}\n'
            f'Número de Target de Treino: {self.target_Treino.shape if self.target_Treino is not None else "Não definido"}\n'
            f'Número de Features de Teste: {self.features_Teste.shape if self.features_Teste is not None else "Não definido"}\n'
            f'Número de Target de Teste: {self.target_Teste.shape if self.target_Teste is not None else "Não definido"}\n'
            f'Pipeline: {self.pipeline if self.pipeline is not None else "Não definido"}'
        )
    
    def padronizarBase(self, pPadronizarNomesDeColunas: bool = True, pSubstituirEspacosColunasPor: str = '_', pPadronizarColunasTexto: bool = True, pSubstituirEspacosTextoPor: str = ' ',
                       pListaDeColunasParaFillNA: list[str] = [], pPreencherNulosCom: str = '') -> None:
        """
        Método para padronizar a base de dados.
        """
        if self.baseDeDados is None:
            raise ValueError("Base de dados não foi carregada. Use o método 'carregarBase' primeiro.")

        # Preencher valores nulos
        if pListaDeColunasParaFillNA:
            for vColuna in pListaDeColunasParaFillNA:
                if vColuna in self.baseDeDados.columns:
                    self.baseDeDados[vColuna] = self.baseDeDados[vColuna].fillna(value=pPreencherNulosCom)
                else:
                    raise ValueError(f"Coluna '{vColuna}' não encontrada na base de dados.")

        # Padronizar nomes de colunas
        if pPadronizarNomesDeColunas:
            self.baseDeDados.columns = [self.__normalizarTexto(pString=vColuna.lower(), pSubstituirEspaco=pSubstituirEspacosColunasPor) for vColuna in self.baseDeDados.columns]

        # Padronizar colunas de texto
        if pPadronizarColunasTexto:
            for vColuna in self.baseDeDados.select_dtypes(include=['object']).columns:
                self.baseDeDados[vColuna] = self.baseDeDados[vColuna].apply(lambda x: self.__normalizarTexto(pString=x.replace('-', ' '), pSubstituirEspaco=pSubstituirEspacosTextoPor) if isinstance(x, str) else x)

    def __normalizarTexto(self, pString: str, pSubstituirEspaco: str = '_') -> str:
        """
            Normaliza um texto removendo caracteres especiais e acentuação, removendo o excesso de espaços e convertendo para maiúsculas.

            Parâmetros:
            - pString (str): A string de entrada a ser normalizada.
            - pSubstituirEspaco (str, opcional): O caractere a ser usado para substituir os espaços. Padrão é '_'.

            Retorna:
            - str: Texto normalizado
        """
        vStringNormalizada = normalize('NFKD', pString)
        vStringNormalizada = sub('[^A-Za-z0-9_ ]','', vStringNormalizada)
        vStringNormalizada = vStringNormalizada.strip().lower().replace(' ', '><').replace('<>', '').replace('><', pSubstituirEspaco)
        vStringNormalizada = ''.join([c for c in vStringNormalizada if not combining(c)])
       
        return vStringNormalizada
    
    def treinarVetorizadorTextual(self, pListaColunas: list[str], pNumeroMaximoFeatures: int = 300) -> None:
        """
        Método para treinar um vetorizador textual usando uma lista de colunas da base de dados (Máximo 2 colunas).
        """
        if self.baseDeDados is None:
            raise ValueError("Base de dados não foi carregada. Use o método 'carregarBase' primeiro.")
        
        if not pListaColunas:
            raise ValueError("Lista de colunas não pode ser vazia.")
        
        if len(pListaColunas) != 2:
            raise ValueError("A lista deve conter exatamente duas colunas para comparar.")
        
        for i in pListaColunas:
            if i not in self.baseDeDados.columns:
                raise ValueError(f"Coluna '{i}' não encontrada na base de dados.")
        
        vListaString = self.baseDeDados[pListaColunas[0]].astype(str).tolist() + self.baseDeDados[pListaColunas[1]].astype(str).tolist()
        
        vVectorizer = TfidfVectorizer(max_features=pNumeroMaximoFeatures)
        vVectorizer.fit(vListaString)

        self.__vVetorizador = vVectorizer

    @property
    def vetorizadorTextual(self) -> TfidfVectorizer:
        """
        Retorna o vetorizador textual treinado.
        """
        if self.__vVetorizador is None:
            raise ValueError("Vetorizador textual não foi treinado. Use o método 'treinarVetorizadorTextual' primeiro.")
        
        return self.__vVetorizador
    
    def salvarVetorizadorTextual(self, pCaminhoArquivo: str) -> None:
        """
        Método para salvar o vetorizador textual treinado em um arquivo.
        """
        if self.__vVetorizador is None:
            raise ValueError("Vetorizador textual não foi treinado. Use o método 'treinarVetorizadorTextual' primeiro.")
        
        joblib.dump(self.vetorizadorTextual, pCaminhoArquivo)

    def carregarVetorizadorTextual(self, pCaminhoArquivo: str) -> None:
        """
        Método para carregar um vetorizador textual de um arquivo.
        """
        try:
            self.__vVetorizador = joblib.load(pCaminhoArquivo)
        except FileNotFoundError:
            raise ValueError(f"Arquivo '{pCaminhoArquivo}' não encontrado.")
        except Exception as e:
            raise ValueError(f"Erro ao carregar o vetorizador textual: {e}")
    
    def calcularSimilaridadeTextual(self, pListaColunas: list[str], pNumeroMaximoFeatures: int = 300) -> pd.array:
        """
        Método para calcular a similaridade textual entre uma lista de strings.
        """
        if self.baseDeDados is None:
            raise ValueError("Base de dados não foi carregada. Use o método 'carregarBase' primeiro.")
        
        if not pListaColunas:
            raise ValueError("Lista de colunas não pode ser vazia.")
        
        if len(pListaColunas) != 2:
            raise ValueError("A lista deve conter exatamente duas colunas para comparar.")
        
        for i in pListaColunas:
            if i not in self.baseDeDados.columns:
                raise ValueError(f"Coluna '{i}' não encontrada na base de dados.")
            
        vListaString_Col1 = self.baseDeDados[pListaColunas[0]].astype(str).tolist()
        vListaString_Col2 = self.baseDeDados[pListaColunas[1]].astype(str).tolist()

        vSimilaridade = np.array(self.vetorizadorTextual.transform(vListaString_Col1).multiply(self.vetorizadorTextual.transform(vListaString_Col2)).sum(axis=1)).ravel()

        return vSimilaridade
    
    def separarFeatureTarget(self, pColunaTarget: str, pColunasIgnorar: list[str] = []) -> None:
        """
        Método para separar a base de dados em features e target.
        """
        if self.baseDeDados is None:
            raise ValueError("Base de dados não foi carregada. Use o método 'carregarBase' primeiro.")
        
        if pColunaTarget not in self.baseDeDados.columns:
            raise ValueError(f"Coluna '{pColunaTarget}' não encontrada na base de dados.")
        
        vListaColunasRemoverDaFeature = [pColunaTarget] + pColunasIgnorar
        vFeatures = self.baseDeDados.drop(columns=vListaColunasRemoverDaFeature)
        vTarget = self.baseDeDados[pColunaTarget]
        
        self.features = vFeatures
        self.target = vTarget

    def criarPipeline(self, pColunasNumericas: list[str], pColunasCategoricas: list[str], pColunasTexto: list[str], pNumeroMaximoFeatures: int) -> None:
        """
        Método para criar uma pipeline de pré-processamento.
        """
        if self.features is None or self.target is None:
            raise ValueError("Features e target não foram separados. Use o método 'separarFeatureTarget' primeiro.")
        
        if not pColunasNumericas and not pColunasCategoricas:
            raise ValueError("Pelo menos uma coluna numérica ou categórica deve ser fornecida.")
        
        if len(pColunasTexto) != 2:
            raise ValueError("A lista de texto deve conter exatamente duas colunas.")
        
        vPipelineNumerico = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]) if pColunasNumericas else None

        vPipelineCategorico = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
        ]) if pColunasCategoricas else None

        vPipelineTexto = TfidfVectorizer(max_features=pNumeroMaximoFeatures)

        vTransformers = []
        if vPipelineNumerico:
            vTransformers.append(('numerical', vPipelineNumerico, pColunasNumericas))
        if vPipelineCategorico:
            vTransformers.append(('categorical', vPipelineCategorico, pColunasCategoricas))
        
        vTransformers.append(('texto1', vPipelineTexto, pColunasTexto[0]))
        vTransformers.append(('texto2', vPipelineTexto, pColunasTexto[1]))

        vPreProcessor = ColumnTransformer(transformers=vTransformers, sparse_threshold=1.0)

        self.pipeline = vPreProcessor

    def salvarPipeline(self, pCaminhoArquivo: str) -> None:
        """
        Método para salvar a pipeline de pré-processamento em um arquivo.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline não foi criada. Use o método 'criarPipeline' primeiro.")
        
        joblib.dump(self.pipeline, pCaminhoArquivo)

    def carregarPipeline(self, pCaminhoArquivo: str) -> None:
        """
        Método para carregar uma pipeline de pré-processamento de um arquivo.
        """
        try:
            self.pipeline = joblib.load(pCaminhoArquivo)
        except FileNotFoundError:
            raise ValueError(f"Arquivo '{pCaminhoArquivo}' não encontrado.")
        except Exception as e:
            raise ValueError(f"Erro ao carregar a pipeline: {e}")

    def separarTreinoTeste(self, pProporcaoTreino: float = 0.2, pRandomState: int = 42) -> None:
        """
        Método para separar a base de dados em conjuntos de treino e teste.
        """
        if self.features is None or self.target is None:
            raise ValueError("Features e target não foram separados. Use o método 'separarFeatureTarget' primeiro.")
        
        if not (0 < pProporcaoTreino < 1):
            raise ValueError("A proporção de treino deve estar entre 0 e 1.")
        
        self.features_Treino, self.features_Teste, self.target_Treino, self.target_Teste = train_test_split(self.features, self.target, test_size=pProporcaoTreino, random_state=pRandomState, stratify=self.target)

    @property
    def features_Treino(self) -> pd.DataFrame:
        """
        Retorna as features do conjunto de treino.
        """
        if self.__vFeatures_Treino is None:
            raise ValueError("Features de treino não foram separadas. Use o método 'separarTreinoTeste' primeiro.")
        
        return self.__vFeatures_Treino
    
    @features_Treino.setter
    def features_Treino(self, pFeatures_Treino: pd.DataFrame):
        """
        Define as features do conjunto de treino.
        """
        if not isinstance(pFeatures_Treino, pd.DataFrame):
            raise TypeError("As features de treino devem ser um DataFrame do pandas.")
        
        self.__vFeatures_Treino = pFeatures_Treino

    @property
    def target_Treino(self) -> pd.Series:
        """
        Retorna o target do conjunto de treino.
        """
        if self.__vTarget_Treino is None:
            raise ValueError("Target de treino não foi separado. Use o método 'separarTreinoTeste' primeiro.")
        
        return self.__vTarget_Treino
    
    @target_Treino.setter
    def target_Treino(self, pTarget_Treino: pd.Series):
        """
        Define o target do conjunto de treino.
        """
        if not isinstance(pTarget_Treino, pd.Series):
            raise TypeError("O target de treino deve ser uma Series do pandas.")
        
        self.__vTarget_Treino = pTarget_Treino

    @property
    def features_Teste(self) -> pd.DataFrame:
        """
        Retorna as features do conjunto de teste.
        """
        if self.__vFeatures_Teste is None:
            raise ValueError("Features de teste não foram separadas. Use o método 'separarTreinoTeste' primeiro.")
        
        return self.__vFeatures_Teste
    
    @features_Teste.setter
    def features_Teste(self, pFeatures_Teste: pd.DataFrame):
        """
        Define as features do conjunto de teste.
        """
        if not isinstance(pFeatures_Teste, pd.DataFrame):
            raise TypeError("As features de teste devem ser um DataFrame do pandas.")
        
        self.__vFeatures_Teste = pFeatures_Teste

    @property
    def target_Teste(self) -> pd.Series:
        """
        Retorna o target do conjunto de teste.
        """
        if self.__vTarget_Teste is None:
            raise ValueError("Target de teste não foi separado. Use o método 'separarTreinoTeste' primeiro.")
        
        return self.__vTarget_Teste
    
    @target_Teste.setter
    def target_Teste(self, pTarget_Teste: pd.Series):
        """
        Define o target do conjunto de teste.
        """
        if not isinstance(pTarget_Teste, pd.Series):
            raise TypeError("O target de teste deve ser uma Series do pandas.")
        
        self.__vTarget_Teste = pTarget_Teste
    
    @property
    def pipeline(self) -> Pipeline:
        """
        Retorna a pipeline de pré-processamento.
        """
        if self.__vPipeline is None:
            raise ValueError("Pipeline não foi criada. Use o método 'criarPipeline' primeiro.")
        
        return self.__vPipeline
    
    @pipeline.setter
    def pipeline(self, pPipeline: Pipeline):
        """
        Define a pipeline de pré-processamento.
        """
        if not isinstance(pPipeline, ColumnTransformer):
            raise TypeError("A pipeline deve ser uma instância de ColumnTransformer do scikit-learn.")
        
        self.__vPipeline = pPipeline

    @property
    def features(self) -> pd.DataFrame:
        """
        Retorna as features separadas da base de dados.
        """
        if self.__vFeatures is None:
            raise ValueError("Features não foram separadas. Use o método 'separarFeatureTarget' primeiro.")
        
        return self.__vFeatures
    
    @property
    def target(self) -> pd.Series:
        """
        Retorna o target separado da base de dados.
        """
        if self.__vTarget is None:
            raise ValueError("Target não foi separado. Use o método 'separarFeatureTarget' primeiro.")
        
        return self.__vTarget
    
    @features.setter
    def features(self, pFeatures: pd.DataFrame):
        """
        Define as features separadas da base de dados.
        """
        if not isinstance(pFeatures, pd.DataFrame):
            raise TypeError("As features devem ser um DataFrame do pandas.")
        
        self.__vFeatures = pFeatures

    @target.setter
    def target(self, pTarget: pd.Series):
        """
        Define o target separado da base de dados.
        """
        if not isinstance(pTarget, pd.Series):
            raise TypeError("O target deve ser uma Series do pandas.")
        
        self.__vTarget = pTarget

    @property
    def modeloXGB(self) -> XGBClassifier:
        """
        Retorna o modelo XGBoost.
        """
        if self.__vModeloXGB is None:
            raise ValueError("Modelo XGBoost não foi criado. Use o método 'treinarModeloXGB' primeiro.")
        
        return self.__vModeloXGB
    
    @modeloXGB.setter
    def modeloXGB(self, pModeloXGB: XGBClassifier):
        """
        Define o modelo XGBoost.
        """
        if not isinstance(pModeloXGB, XGBClassifier):
            raise TypeError("O modelo XGBoost deve ser uma instância de XGBClassifier do xgboost.")
        
        self.__vModeloXGB = pModeloXGB

    def executarPipeline(self, pAplicarEm: str = 'Treino_Teste') -> None:
        """
        Executa a pipeline de pré-processamento nas features fornecidas.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline não foi criada. Use o método 'criarPipeline' primeiro.")
        
        if pAplicarEm.upper() not in ['TREINO', 'TESTE', 'TREINO_TESTE']:
            raise ValueError("O parâmetro 'pAplicarEm' deve ser 'Treino', 'Teste' ou 'Treino_Teste'.")
        
        if pAplicarEm.upper() == 'TREINO':
            if self.features_Treino is None:
                raise ValueError("Features de treino não foram definidas. Use o método 'separarTreinoTeste' primeiro.")
            self.features_Treino_Transformadas = self.pipeline.fit_transform(self.features_Treino)
        elif pAplicarEm.upper() == 'TESTE':
            if self.features_Teste is None:
                raise ValueError("Features de teste não foram definidas. Use o método 'separarTreinoTeste' primeiro.")
            self.features_Teste_Transformadas = self.pipeline.transform(self.features_Teste)
        else:
            if self.features_Treino is None or self.features_Teste is None:
                raise ValueError("Features de treino e teste não foram definidas. Use o método 'separarTreinoTeste' primeiro.")
            self.features_Treino_Transformadas = self.pipeline.fit_transform(self.features_Treino)
            self.features_Teste_Transformadas = self.pipeline.transform(self.features_Teste)

    @property
    def features_Treino_Transformadas(self) -> np.array:
        """
        Retorna as features de treino transformadas pela pipeline.
        """
        if self.__vFeatures_Treino_Transformadas is None:
            raise ValueError("Features de treino transformadas não foram criadas. Use o método 'executarPipeline' primeiro.")
        
        return self.__vFeatures_Treino_Transformadas
    
    @features_Treino_Transformadas.setter
    def features_Treino_Transformadas(self, pFeatures_Treino_Transformadas: np.array):
        """
        Define as features de treino transformadas pela pipeline.
        """
        self.__vFeatures_Treino_Transformadas = pFeatures_Treino_Transformadas
    
    @property
    def features_Teste_Transformadas(self) -> np.array:
        """
        Retorna as features de teste transformadas pela pipeline.
        """
        if self.__vFeatures_Teste_Transformadas is None:
            raise ValueError("Features de teste transformadas não foram criadas. Use o método 'executarPipeline' primeiro.")
        
        return self.__vFeatures_Teste_Transformadas
    
    @features_Teste_Transformadas.setter
    def features_Teste_Transformadas(self, pFeatures_Teste_Transformadas: np.array):
        """
        Define as features de teste transformadas pela pipeline.
        """
        self.__vFeatures_Teste_Transformadas = pFeatures_Teste_Transformadas

    def criarModeloXGB(self, pParametros: dict = None, pTreinarModelo: bool = False) -> None:
        """
        Método para criar e treinar um modelo XGBoost.
        """
        if self.features_Treino_Transformadas is None or self.target_Treino is None:
            raise ValueError("Features de treino transformadas e target de treino não foram definidos. Use os métodos 'executarPipeline' e 'separarTreinoTeste' primeiro.")
        
        vParametros = pParametros if pParametros else {}
        self.modeloXGB = XGBClassifier(**vParametros)

        if pTreinarModelo:
            self.treinarModeloXGB()


    def treinarModeloXGB(self) -> None:
        """
        Método para treinar o modelo XGBoost com as features e target de treino.
        """
        if self.modeloXGB is None:
            raise ValueError("Modelo XGBoost não foi criado. Use o método 'criarModeloXGB' primeiro.")
        
        if self.features_Treino_Transformadas is None or self.target_Treino is None:
            raise ValueError("Features de treino transformadas e target de treino não foram definidos. Use os métodos 'executarPipeline' e 'separarTreinoTeste' primeiro.")
        
        self.modeloXGB.fit(self.features_Treino_Transformadas, self.target_Treino, eval_set=[(self.features_Teste_Transformadas, self.target_Teste)], verbose=False)

    def avaliarModeloXGB(self, pOutputDict: bool = False) -> str:
        """
        Método para avaliar o modelo XGBoost usando o conjunto de teste.
        Retorna um relatório de classificação.
        """
        if self.modeloXGB is None:
            raise ValueError("Modelo XGBoost não foi criado. Use o método 'criarModeloXGB' primeiro.")
        
        if self.features_Teste_Transformadas is None or self.target_Teste is None:
            raise ValueError("Features de teste transformadas e target de teste não foram definidos. Use os métodos 'executarPipeline' e 'separarTreinoTeste' primeiro.")
        
        vPredicoes = self.modeloXGB.predict(self.features_Teste_Transformadas)
        vRelatorio = classification_report(self.target_Teste, vPredicoes, output_dict=pOutputDict, digits=3)
        
        return vRelatorio
    
    def salvarModeloXGB(self, pCaminhoArquivo: str) -> None:
        """
        Método para salvar o modelo XGBoost em um arquivo.
        """
        if self.modeloXGB is None:
            raise ValueError("Modelo XGBoost não foi criado. Use o método 'criarModeloXGB' primeiro.")
        
        joblib.dump(self.modeloXGB, pCaminhoArquivo)

    def carregarModeloXGB(self, pCaminhoArquivo: str) -> None:
        """
        Método para carregar um modelo XGBoost de um arquivo.
        """
        try:
            self.modeloXGB = joblib.load(pCaminhoArquivo)
        except FileNotFoundError:
            raise ValueError(f"Arquivo '{pCaminhoArquivo}' não encontrado.")
        except Exception as e:
            raise ValueError(f"Erro ao carregar o modelo XGBoost: {e}")
        

    def prever(self, pFeatures: pd.DataFrame) -> np.array:
        """
        Método para fazer previsões usando o modelo XGBoost.
        """
        if self.modeloXGB is None:
            raise ValueError("Modelo XGBoost não foi criado. Use o método 'criarModeloXGB' primeiro.")
        
        if self.pipeline is None:
            raise ValueError("Pipeline não foi criada. Use o método 'criarPipeline' primeiro.")
        
        if not isinstance(pFeatures, pd.DataFrame):
            raise TypeError("As features devem ser um DataFrame do pandas.")
        
        vFeatures_Transformadas = self.pipeline.transform(pFeatures)
        vPredicoes = self.modeloXGB.predict(vFeatures_Transformadas)
        
        return vPredicoes

    def preverProbabilidades(self, pFeatures: pd.DataFrame) -> np.array:
        """
        Método para prever as probabilidades usando o modelo XGBoost.
        """
        if self.modeloXGB is None:
            raise ValueError("Modelo XGBoost não foi criado. Use o método 'criarModeloXGB' primeiro.")
        
        if self.pipeline is None:
            raise ValueError("Pipeline não foi criada. Use o método 'criarPipeline' primeiro.")
        
        if not isinstance(pFeatures, pd.DataFrame):
            raise TypeError("As features devem ser um DataFrame do pandas.")
        
        vFeatures_Transformadas = self.pipeline.transform(pFeatures)
        vProbabilidades = self.modeloXGB.predict_proba(vFeatures_Transformadas)[:, 1]
        
        return vProbabilidades

    def __del__(self):
        """
        Método chamado quando o objeto é destruído.
        Limpa a base de dados carregada.
        """
        self.__vBase = None
        self.caminhoBase = ''