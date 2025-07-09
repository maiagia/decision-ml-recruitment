import os
import json
import pandas as pd
import re

def padronizar_nivel(texto, mapa_niveis):
    if not texto or not isinstance(texto, str):
        return "Desconhecido"
    texto = texto.strip().lower()
    for chave, valor in mapa_niveis.items():
        if chave in texto:
            return valor
    return "Outro"

def limpar_texto(texto):
    if not texto or not isinstance(texto, str):
        return ""
    return re.sub(r"\s+", " ", texto).strip()

def verificar_aprovacao(situacao):
    if not situacao or not isinstance(situacao, str):
        return 0

    situacao = situacao.strip().lower()

    # Lista das situações que indicam contratação ou aprovação
    situacoes_positivas = {
        "aprovado",
        "proposta aceita",
        "contratado como hunting",
        "contratado pela decision"
    }

    return 1 if situacao in situacoes_positivas else 0

# Mapas de padronização
mapa_nivel_ingles = {
    "básico": "Básico", "intermediário": "Intermediário",
    "avançado": "Avançado", "fluente": "Fluente",
    "não informado": "Desconhecido", "desconhecido": "Desconhecido"
}
mapa_nivel_espanhol = mapa_nivel_ingles
mapa_nivel_academico = {
    "fundamental": "Fundamental", "médio": "Médio", "superior": "Superior",
    "pós": "Pós-graduação", "mestrado": "Mestrado", "doutorado": "Doutorado",
    "não informado": "Desconhecido", "desconhecido": "Desconhecido"
}

# Carregamento dos dados
with open("../data/vagas.json", encoding="utf-8") as f:
    jobs_data = json.load(f)

with open("../data/applicants.json", encoding="utf-8") as f:
    applicants_data = json.load(f)

with open("../data/prospects.json", encoding="utf-8") as f:
    prospects_data = json.load(f)

registros = []

# Processamento principal
for vaga_id, prospect_info in prospects_data.items():
    vaga = jobs_data.get(vaga_id)
    if not vaga:
        continue

    for prospect in prospect_info.get("prospects", []):
        codigo_candidato = prospect.get("codigo")
        candidato = applicants_data.get(codigo_candidato, {})

        perfil = vaga.get("perfil_vaga", {})
        tecnicos = perfil.get("competencia_tecnicas_e_comportamentais", "")
        atividades = perfil.get("principais_atividades", "")
        requisitos = limpar_texto(f"{tecnicos} {atividades}")

        info_formacao = candidato.get("formacao_e_idiomas", {})
        info_profissional = candidato.get("informacoes_profissionais", {})
        cv_raw = candidato.get("cv_pt", "")

        nivel_ingles_vaga = padronizar_nivel(perfil.get("nivel_ingles"), mapa_nivel_ingles)
        nivel_espanhol_vaga = padronizar_nivel(perfil.get("nivel_espanhol"), mapa_nivel_espanhol)
        nivel_academico = padronizar_nivel(info_formacao.get("nivel_academico"), mapa_nivel_academico)
        nivel_ingles = padronizar_nivel(info_formacao.get("nivel_ingles"), mapa_nivel_ingles)
        nivel_espanhol = padronizar_nivel(info_formacao.get("nivel_espanhol"), mapa_nivel_espanhol)

        cv_texto = limpar_texto(cv_raw)
        conhecimentos_tecnicos = limpar_texto(info_profissional.get("conhecimentos_tecnicos", ""))

        titulo_vaga = vaga.get("informacoes_basicas", {}).get("titulo_vaga", "Desconhecido")
        cliente = vaga.get("informacoes_basicas", {}).get("cliente", "Desconhecido")
        nivel_vaga = perfil.get("nivel profissional") or "Desconhecido"
        local_vaga = perfil.get("cidade") or "Desconhecido"

        nome_candidato = candidato.get("informacoes_pessoais", {}).get("nome", "Desconhecido")
        situacao = prospect.get("situacao_candidado", "Não informado")
        comentario = prospect.get("comentario", "")
        recrutador = prospect.get("recrutador", "")

        registros.append({
            "vaga_id": vaga_id,
            "titulo_vaga": titulo_vaga,
            "cliente": cliente,
            "nivel_vaga": nivel_vaga,
            "nivel_ingles_vaga": nivel_ingles_vaga,
            "nivel_espanhol_vaga": nivel_espanhol_vaga,
            "local_vaga": local_vaga,
            "requisitos_vaga": requisitos,

            "codigo_candidato": codigo_candidato if codigo_candidato else "Desconhecido",
            "nome_candidato": nome_candidato,
            "nivel_academico": nivel_academico,
            "nivel_ingles": nivel_ingles,
            "nivel_espanhol": nivel_espanhol,
            "conhecimentos_tecnicos": conhecimentos_tecnicos,
            "cv_texto": cv_texto,

            "nivel_profissional": info_profissional.get("nivel_profissional", "Desconhecido"),
            "local_candidato": info_profissional.get("cidade", "Desconhecido"),

            "situacao": situacao,
            "comentario": comentario,
            "recrutador": recrutador
        })

# Cria DataFrame consolidado
df = pd.DataFrame(registros)

# Criação da coluna match
df["match"] = df["situacao"].apply(verificar_aprovacao)

# Verificação da distribuição do target
total_registros = len(df)
contagem_match = df['match'].value_counts()
aprovados = contagem_match.get(1, 0)
nao_aprovados = contagem_match.get(0, 0)

print(f"\nTotal de registros: {total_registros}")
print(f"Quantidade de candidatos aprovados (match=1): {aprovados}")
print(f"Quantidade de candidatos não aprovados (match=0): {nao_aprovados}")

if total_registros > 0:
    print(f"Aprovados (%): {aprovados / total_registros * 100:.2f}%")
    print(f"Não aprovados (%): {nao_aprovados / total_registros * 100:.2f}%")
else:
    print("O DataFrame está vazio.")

# Cria pasta de saída se não existir
output_path = os.path.abspath("../output")
os.makedirs(output_path, exist_ok=True)

# Salva o DataFrame como CSV
output_file = os.path.join(output_path, "dataset_unificado.csv")
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"\n✅ Dataset salvo com {len(df)} registros em: {output_file}")

# Schema e amostra
print("\n📋 Schema do DataFrame (colunas e tipos de dados):")
print(df.dtypes)

print("\n📊 Primeiras 100 linhas do dataset:")
print(df.head(100))