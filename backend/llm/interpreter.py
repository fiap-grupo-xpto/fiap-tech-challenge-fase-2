import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def classify_risk(probability: float) -> str:
    if probability >= 0.7:
        return "alto"
    if probability >= 0.4:
        return "moderado"
    return "baixo"


def _generate_interpretation(prompt: str, model_name: str, probability: float, risk_level: str) -> str:
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        return response.text

    except Exception as e:
        return (
            f"O modelo {model_name} indicou uma probabilidade estimada de "
            f"{probability * 100:.2f}%, classificada como risco {risk_level}. "
            "Não foi possível gerar a interpretação pela LLM neste momento. "
            f"Detalhes técnicos: {str(e)}"
        )


def generate_tabular_interpretation(result: dict, model_name: str, threshold: float) -> str:
    probability = result["probability"]
    disease_detected = result["disease_detected"]
    risk_level = classify_risk(probability)

    prompt = f"""
Você é uma LLM integrada a um sistema de apoio ao diagnóstico médico.

Explique o resultado abaixo de forma clara, objetiva e responsável.

Dados do modelo:
- Modelo utilizado: {model_name}
- Tipo de exame: triagem tabular a partir de dados demográficos e sintomas informados pelo paciente
- Probabilidade estimada: {probability * 100:.2f}%
- Limiar de decisão: {threshold * 100:.2f}%
- Doença detectada pelo modelo: {"sim" if disease_detected else "não"}
- Nível de risco estimado: {risk_level}

Responda obrigatoriamente neste formato:

Resumo:
Risco:
Justificativa:
Recomendação:
Observação:

Regras:
- Não diga que o paciente tem diagnóstico confirmado.
- Não substitua avaliação médica.
- Use linguagem adequada para profissionais da saúde.
- Seja objetivo.
- A recomendação deve ser compatível com o nível de risco.
- A observação deve deixar claro que o resultado é apoio à decisão clínica.
- Não use markdown, negrito, asteriscos ou listas.
"""

    return _generate_interpretation(prompt, model_name, probability, risk_level)


def generate_image_interpretation(result: dict, model_name: str, threshold: float) -> str:
    probability = result["probability"]
    disease_detected = result["disease_detected"]
    risk_level = classify_risk(probability)

    prompt = f"""
Você é uma LLM integrada a um sistema de apoio ao diagnóstico médico.

Explique o resultado abaixo de forma clara, objetiva e responsável.

Dados do modelo:
- Modelo utilizado: {model_name}
- Tipo de exame: análise de imagem de ressonância por visão computacional
- Probabilidade estimada: {probability * 100:.2f}%
- Limiar de decisão: {threshold * 100:.2f}%
- Achado suspeito detectado pelo modelo: {"sim" if disease_detected else "não"}
- Nível de risco estimado: {risk_level}

Responda obrigatoriamente neste formato:

Resumo:
Risco:
Justificativa:
Recomendação:
Observação:

Regras:
- Não diga que o paciente tem diagnóstico confirmado.
- Não substitua avaliação médica nem laudo radiológico.
- Use linguagem adequada para profissionais da saúde.
- Seja objetivo.
- A recomendação deve ser compatível com o nível de risco.
- A observação deve deixar claro que o resultado é apoio à decisão clínica, baseado apenas na análise automática da imagem.
- Não use markdown, negrito, asteriscos ou listas.
"""

    return _generate_interpretation(prompt, model_name, probability, risk_level)