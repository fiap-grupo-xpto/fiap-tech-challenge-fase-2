import base64
import io
import os
from pathlib import Path

import pandas as pd
pd.set_option("mode.string_storage", "python")
import requests
import streamlit as st


st.set_page_config(layout="wide", page_title="Tech Challenge Fase 01")
API_URL = os.getenv("API_URL", "http://localhost:8888")

EXPECTED_COLUMNS = [
    "GENDER",
    "AGE",
    "SMOKING",
    "YELLOW_FINGERS",
    "ANXIETY",
    "PEER_PRESSURE",
    "CHRONIC_DISEASE",
    "FATIGUE",
    "ALLERGY",
    "WHEEZING",
    "ALCOHOL_CONSUMING",
    "COUGHING",
    "SHORTNESS_OF_BREATH",
    "SWALLOWING_DIFFICULTY",
    "CHEST_PAIN",
]


static_dir = Path(__file__).parent / "static"
logo_path = static_dir / "logo.png"

try:
    logo_bytes = logo_path.read_bytes()
    logo_base64 = base64.b64encode(logo_bytes).decode("utf-8")
    st.markdown(
        f"""
        <div style="display:flex; justify-content:center; margin-bottom:2rem;">
            <img src="data:image/png;base64,{logo_base64}" style="max-width:200px; width:100%; height:auto;" />
        </div>
        """,
        unsafe_allow_html=True,
    )
except FileNotFoundError:
    st.error("Logo não encontrado em: " + str(logo_path))


if "stage" not in st.session_state:
    st.session_state.stage = 1
if "image_key" not in st.session_state:
    st.session_state.image_key = 0
if "csv_page_index" not in st.session_state:
    st.session_state.csv_page_index = 0
if "csv_page_size" not in st.session_state:
    st.session_state.csv_page_size = 50
if "csv_results_display_df" not in st.session_state:
    st.session_state.csv_results_display_df = None
if "csv_results_total_positive" not in st.session_state:
    st.session_state.csv_results_total_positive = 0
if "csv_results_total_negative" not in st.session_state:
    st.session_state.csv_results_total_negative = 0
if "csv_results_total_rows" not in st.session_state:
    st.session_state.csv_results_total_rows = 0


def go_to_stage(stage: int) -> None:
    st.session_state.stage = stage
    st.rerun()


st.title("Tech Challenge Fase 01")
st.subheader("FIAP - IA para devs - Grupo 69")
st.write(
    "O fluxo foi dividido em duas etapas: primeiro a pré-triagem por CSV e, depois, a análise de imagens de ressonância."
)
st.warning(
    "⚠️ Lembre-se: este é um modelo de apoio à triagem e pode errar. A decisão final deve sempre ser tomada por um(a) profissional de saúde."
)


if st.session_state.stage == 1:
    st.header("Etapa 1 - Pré-triagem inicial com CSV")
    st.write(
        "Envie um arquivo CSV com as mesmas colunas usadas no treinamento. O sistema vai retornar uma previsão por linha em forma de tabela."
    )

    with st.expander("Ver colunas esperadas", expanded=False):
        st.code("\n".join(EXPECTED_COLUMNS))

    uploaded_csv = st.file_uploader(
        "Faça o upload do CSV para pré-triagem:",
        type=["csv"],
        key="csv_uploader",
    )

    col_run, col_next = st.columns([0.9, 1.6], gap="small")
    with col_run:
        run_csv = st.button(
            "Executar pré-triagem",
            type="primary",
            disabled=uploaded_csv is None,
            use_container_width=True,
        )
    with col_next:
        if st.button(
            "Ir para segunda etapa de ressonância",
            type="secondary",
            use_container_width=True,
        ):
            go_to_stage(2)

    if run_csv and uploaded_csv is not None:
        try:
            csv_bytes = uploaded_csv.getvalue()
            input_df = pd.read_csv(io.BytesIO(csv_bytes))

            request_url = f"{API_URL}/analyze-tabular"
            with st.spinner("Enviando CSV para análise..."):
                response = requests.post(
                    request_url,
                    files={
                        "csv_file": (
                            uploaded_csv.name,
                            csv_bytes,
                            uploaded_csv.type or "text/csv",
                        )
                    },
                    timeout=120,
                )

            if response.status_code != 200:
                st.error(f"Erro ao enviar CSV para análise. Status code: {response.status_code}")
            else:
                result = response.json()
                if result.get("status") != "success":
                    st.error(result.get("message", "Erro desconhecido ao analisar o CSV."))
                else:
                    prediction_df = pd.DataFrame(result.get("results", []))
                    if prediction_df.empty:
                        st.warning("A API retornou sucesso, mas não trouxe resultados.")
                    else:
                        display_df = input_df.reset_index(drop=True).copy()
                        if len(display_df) == len(prediction_df):
                            display_df.insert(0, "Linha", display_df.index + 1)
                            display_df["Resultado da triagem"] = prediction_df["disease_detected"].map(
                                {True: "Positivo", False: "Negativo"}
                            )
                            display_df["Probabilidade (%)"] = (prediction_df["probability"] * 100).round(2)
                            display_df["Interpretação da LLM"] = prediction_df["llm_interpretation"]
                        else:
                            display_df = prediction_df.copy()
                            display_df.insert(0, "Linha", display_df.index + 1)
                            display_df["Probabilidade (%)"] = (display_df["probability"] * 100).round(2)
                            display_df["Interpretação da LLM"] = display_df["llm_interpretation"]

                        st.session_state.csv_results_display_df = display_df
                        st.session_state.csv_results_total_rows = len(prediction_df)
                        st.session_state.csv_results_total_positive = int(prediction_df["disease_detected"].sum())
                        st.session_state.csv_results_total_negative = (
                            st.session_state.csv_results_total_rows - st.session_state.csv_results_total_positive
                        )
                        st.session_state.csv_page_index = 0

        except Exception as e:
            st.error(f"Erro durante a análise do CSV: {e}")

    if st.session_state.csv_results_display_df is not None:
        display_df = st.session_state.csv_results_display_df
        total_positive = st.session_state.csv_results_total_positive
        total_negative = st.session_state.csv_results_total_negative
        total_rows = st.session_state.csv_results_total_rows

        summary_col1, summary_col2, summary_col3 = st.columns(3)
        summary_col1.metric("Linhas analisadas", total_rows)
        summary_col2.metric("Casos suspeitos", total_positive)
        summary_col3.metric("Casos negativos", total_negative)

        if total_positive > 0:
            st.warning(
                f"⚠️ Foram identificados {total_positive} casos suspeitos no CSV. Isso é uma triagem inicial, não um diagnóstico final."
            )
        else:
            st.success("✨ Nenhum caso suspeito foi identificado na triagem inicial.")

        llm_column = "Interpretação da LLM"
        if llm_column in display_df.columns:
            interpreted_rows = display_df[
                ~display_df[llm_column].fillna("").str.startswith("Interpretação por LLM não gerada")
            ]
            if not interpreted_rows.empty:
                st.subheader("🩺 Interpretação da LLM")
                st.caption(
                    f"Explicações em linguagem natural geradas pela LLM para os {len(interpreted_rows)} "
                    "primeiros casos analisados."
                )
                for _, row in interpreted_rows.iterrows():
                    resultado = row.get("Resultado da triagem", "-")
                    probabilidade = row.get("Probabilidade (%)", 0)
                    with st.expander(f"Linha {int(row['Linha'])} — {resultado} ({probabilidade:.2f}%)"):
                        st.text(row[llm_column])

        st.caption("Linhas marcadas em vermelho indicam casos suspeitos e devem ser priorizadas para revisão.")

        def highlight_positive_rows(row):
            is_positive = row.get("Resultado da triagem") == "Positivo"
            style = "background-color: #f3c9c9; color: #000000;" if is_positive else ""
            return [style] * len(row)

        st.session_state.csv_page_size = st.selectbox(
            "Linhas por página",
            [25, 50, 100],
            index=[25, 50, 100].index(st.session_state.csv_page_size)
            if st.session_state.csv_page_size in [25, 50, 100]
            else 1,
        )

        total_pages = max((total_rows - 1) // st.session_state.csv_page_size + 1, 1)
        st.session_state.csv_page_index = min(
            st.session_state.csv_page_index,
            total_pages - 1,
        )
        start_row = st.session_state.csv_page_index * st.session_state.csv_page_size
        end_row = min(start_row + st.session_state.csv_page_size, total_rows)
        page_df = display_df.iloc[start_row:end_row].drop(columns=[llm_column], errors="ignore").copy()

        nav_prev, nav_info, nav_next = st.columns([0.55, 1.1, 2.35])
        with nav_prev:
            prev_left, prev_right = st.columns([0.92, 0.08])
            with prev_left:
                if st.button(
                    "⬅️ Anterior",
                    disabled=st.session_state.csv_page_index == 0,
                    key="csv_prev_page",
                ):
                    st.session_state.csv_page_index = max(st.session_state.csv_page_index - 1, 0)
                    st.rerun()
        with nav_info:
            st.markdown(
                f"<div style='text-align:center; color: rgba(250,250,250,0.65); padding-top: 0.5rem;'>"
                f"Mostrando {start_row + 1}-{end_row} de {total_rows} linhas"
                f"</div>",
                unsafe_allow_html=True,
            )
        with nav_next:
            next_left, next_right = st.columns([0.1, 0.9])
            with next_right:
                if st.button(
                    "Próxima ➡️",
                    disabled=st.session_state.csv_page_index >= total_pages - 1,
                    key="csv_next_page",
                ):
                    st.session_state.csv_page_index = min(
                        st.session_state.csv_page_index + 1,
                        total_pages - 1,
                    )
                    st.rerun()

        st.dataframe(
            page_df.style.apply(highlight_positive_rows, axis=1),
            use_container_width=True,
        )

        st.download_button(
            "Baixar resultado da triagem",
            data=display_df.to_csv(index=False).encode("utf-8"),
            file_name="triagem_cancer_pulmao.csv",
            mime="text/csv",
        )


if st.session_state.stage == 2:
    st.header("Etapa 2 - Análise de imagens de ressonância")
    st.write(
        "Nesta etapa você pode enviar imagens de ressonância para a análise automatizada do modelo de visão computacional."
    )

    if st.button("⬅️ Voltar para a triagem por CSV", type="secondary"):
        go_to_stage(1)

    st.info(
        "O limite inicial vem do treino do modelo e pode ser ajustado manualmente se necessário."
    )
    probability_threshold = st.number_input(
        "",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=0.01,
        format="%.2f",
        width=150,
        label_visibility="collapsed",
    )

    uploaded_files = st.file_uploader(
        "Faça o upload das imagens de ressonância para análise:",
        accept_multiple_files=True,
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        key=f"image_uploader_{st.session_state.image_key}",
    )

    col_run, col_clear, col_spacer = st.columns([0.55, 0.55, 1.9])
    with col_run:
        run_images = st.button(
            "Executar análise de imagens",
            type="primary",
            disabled=not uploaded_files,
        )
    with col_clear:
        if st.button("🗑️ Limpar imagens", type="secondary"):
            st.session_state.image_key += 1
            st.rerun()

    if run_images and uploaded_files:
        request_url = f"{API_URL}/analyze-images"
        files = []
        for item in uploaded_files:
            file_content = item.getvalue()
            files.append(("files", (item.name, io.BytesIO(file_content), item.type)))

        with st.spinner("Enviando arquivos para análise..."):
            try:
                res = requests.post(
                    request_url,
                    files=files,
                    data={"probability_threshold": probability_threshold / 100.0},
                    timeout=120,
                )
                if res.status_code == 200:
                    result = res.json()
                    count = len(files)
                    st.subheader(
                        f"Resultados ({count} {'imagem analisada' if count == 1 else 'imagens analisadas'})"
                    )
                    total_disease = sum(1 for r in result.get("results", []) if r.get("disease_detected"))
                    if total_disease > 0:
                        st.warning(
                            f"⚠️ Total de imagens com achado suspeito: {total_disease} de {count}. Consulte um profissional de saúde para avaliação detalhada."
                        )
                    else:
                        st.success(f"✨ Nenhuma anomalia detectada em {count} imagens.")

                    if result.get("results"):
                        for file, r in zip(uploaded_files, result["results"]):
                            col_img, col_res = st.columns(2)
                            with col_img:
                                st.image(file, caption="Imagem analisada", width="content")
                            with col_res:
                                if r.get("status") == "success":
                                    st.write(f"**Nome do arquivo:** {file.name}")
                                    st.write(f"**Tipo do arquivo:** {file.type}")
                                    st.write(f"**Tamanho do arquivo:** {file.size} bytes")
                                    st.write(
                                        f"**Limiar usado:** {float(r.get('threshold_used', probability_threshold / 100.0)) * 100:.2f}%"
                                    )
                                    st.write(f"🔍 Achado suspeito: {'Sim' if r.get('disease_detected') else 'Não'}")
                                    st.write(f"📈 Probabilidade: {float(r.get('probability', 0)) * 100:.2f}%")
                                    if r.get("disease_detected"):
                                        st.warning("⚠️ Recomendada avaliação médica detalhada")
                                    else:
                                        st.success("✨ Nenhuma anomalia detectada")

                                    llm_interpretation = r.get("llm_interpretation", "")
                                    if llm_interpretation and not llm_interpretation.startswith(
                                        "Interpretação por LLM não gerada"
                                    ):
                                        with st.expander("🩺 Interpretação da LLM"):
                                            st.text(llm_interpretation)
                                else:
                                    st.write("❌ Status: Erro")
                                    st.write(f"Mensagem: {r.get('message')}")
                            st.write("---")
                else:
                    st.error(f"Erro ao enviar arquivos para análise. Status code: {res.status_code}")
            except Exception as e:
                st.error(f"Erro durante a análise: {str(e)}")
