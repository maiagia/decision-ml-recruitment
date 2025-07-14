from pydantic import BaseModel
from typing import List

class ModeloRequestPreco(BaseModel):
    """
    Modelo para a predição de preços futuros.
    """
    precos_anteriores: List[float]