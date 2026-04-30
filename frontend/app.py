import streamlit as st
import requests
import io
import os
from pathlib import Path

st.set_page_config(layout="centered", page_title="Tech Challenge Fase 01")
API_URL = os.getenv("API_URL", "http://localhost:8888")

# Get the absolute path to the static directory
static_dir = Path(__file__).parent / "static"
logo_path = static_dir / "logo.png"

# Read the logo image as bytes
with open(logo_path, "rb") as f:
    logo_contents = f.read()

# Add custom CSS to center the logo
st.markdown(
    """
    <style>
        .logo-container {
            display: flex;
            justify-content: center;
            margin-bottom: 2rem;
        }
        .logo-container img {
            max-width: 200px;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Logo and title in a container using st.image
col1, col2, col3 = st.columns([1,2,1])
with col2:
    try:
        st.image(logo_path, width=200)
    except FileNotFoundError:
        st.error("Logo não encontrado em: " + str(logo_path))


st.title("Tech Challenge Fase 01")
st.subheader("FIAP - IA para devs - Grupo 71")

# Initialize session state for file uploader key if not exists
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0
if 'dialog_open' not in st.session_state:
    st.session_state.dialog_open = False

st.write("Este aplicativo permite o upload de imagens de exames de raio-x para análise de tuberculose utilizando um modelo de Machine Learning.")

st.warning("⚠️ Lembre-se: este é um modelo de Machine Learning e pode não ser 100% preciso. Sempre consulte um profissional de saúde para diagnóstico definitivo.")

st.info("Defina no campo abaixo o limite de probabilidade para classificar uma imagem como positiva para tuberculose. Recomendado: 91%.")
probability_threshold = st.number_input("", min_value=0, max_value=100, value=91, step=1, width=150, label_visibility="collapsed")

uploaded_files = st.file_uploader(
    "Faça o upload das imagens de exames raio-x para análise:",
    accept_multiple_files=True,
    type=['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'], # Specify accepted file types
    key=f"file_uploader_{st.session_state.uploader_key}",
)

if uploaded_files:
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]
        
    if (st.session_state.get('last_uploader_key') != st.session_state.uploader_key):
        st.session_state.last_uploader_key = st.session_state.uploader_key

    # Open a confirmation dialog when the user clicks the clear button
    if st.button("🗑️ Limpar resultados", type="secondary", key="show_confirm_btn"):
        # mark dialog open and avoid running the analysis in this run
        st.session_state.uploader_key += 1
        st.rerun()
        

    analysis_header = st.empty()  # Create a placeholder for the header
    analysis_header.subheader("Analisando imagens...")
    
    
    # Prepare all files for a single request
    request_url = f"{API_URL}/analyze-images"
    # Build `files` payload from the persisted `files_source` so it works
    # whether files_source contains UploadFile objects or our saved dicts.
    files = []
    for item in uploaded_files:
        # Read the file content into BytesIO
        file_content = item.read()
        # Reset the file pointer for potential future reads
        item.seek(0)
        files.append(("files", (item.name, io.BytesIO(file_content), item.type)))
    
    with st.spinner("Enviando arquivos para análise..."):
        try:
            res = requests.post(request_url, files=files, data={"probability_threshold": probability_threshold / 100.0})
            if res.status_code == 200:
                result = res.json()
                count = len(files)
                analysis_header.subheader(f"Resultados ({count} {'imagem analisada' if count == 1 else 'imagens analisadas'})")
                # coloque na msg abaixo o total de r['disease_detected'] == True
                total_disease = sum(1 for r in result.get("results", []) if r.get("disease_detected"))
                if total_disease > 0:
                    st.warning(f"⚠️ Total de imagens com tuberculose detectada: {total_disease} de {count}. Consulte um profissional de saúde para avaliação detalhada.")
                else:
                    st.success(f"✨ Nenhuma anomalia detectada em {count} imagens.")

                # Display results for each file
                if result.get("results"):
                    for file, r in zip(uploaded_files, result["results"]):
                        
                        # Create two columns: one for image, one for results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Show image from the uploaded file
                            st.image(file, caption="Imagem analisada", width="content")
                        
                        with col2:
                            if r['status'] == 'success':
                                st.write(f"**Nome do arquivo:** {file.name}")
                                st.write(f"**Tipo do arquivo:** {file.type}")
                                st.write(f"**Tamanho do arquivo:** {file.size} bytes")
                                st.write(f"🔍 Tuberculose detectada: {'Sim' if r['disease_detected'] else 'Não'}")
                                st.write(f"📈 Probabilidade: {float(r['probability']) * 100:.2f}%")
                                if r['disease_detected']:
                                    st.warning("⚠️ Recomendada avaliação médica detalhada")
                                else:
                                    st.success("✨ Nenhuma anomalia detectada")
                            else:
                                st.write("❌ Status: Erro")
                                st.write(f"Mensagem: {r.get('message')}")
                        
                        st.write("---")
            else:
                st.error(f"Erro ao enviar arquivos para análise. Status code: {res.status_code}")
        except Exception as e:
            st.error(f"Erro durante a análise: {str(e)}")
else:
    st.info("É necessário fazer o upload de pelo menos uma imagem de raio-x para análise.")
