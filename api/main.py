# from fastapi import FastAPI
# from rotas import predict, historico_preco

# vApp = FastAPI(title="ML Recruitment")

# vApp.include_router(router=predict.vRota, prefix='/api')
# vApp.include_router(router=historico_preco.vRota, prefix='/api')


from classes.ml_recruitment import ML_Recruitment
import os

vML = ML_Recruitment(pCaminhoBase=os.path.join("..", "data", "base.csv"))
# vML.carregarBase()

print(ML_Recruitment.caminhoBase)
