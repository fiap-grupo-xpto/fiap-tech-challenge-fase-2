# FIAP - IA PARA DEVS - Tech Challenge Fase 01 - Grupo XPTO (Grupo 69)

# Integrantes do Grupo 69

Otoniel da Silva Isidoro - rm368069

André Roberto Figueiró de Magalhães - rm365608

Lucas de Melo Lima - rm373930

Gustavo César de Souza - rm370800

Thales Ernane de Souza - rm372083

# Overview
Este projeto tem o intuito de treinar modelos para análise de risco de cancer de pulmão através de dados demográficos de pacientes (dados tabulares csv) e detecção de cancer de pulmão através de imagens de ressonância (visão computacional).

# Vídeo de demonstração: 

[![(7) FIAP - Grupo 69 - Tech Challenge - Fase 1 - YouTube](https://i.ytimg.com/vi/-SdZHwdxZpI/hqdefault.jpg)](https://www.youtube.com/watch?v=-SdZHwdxZpI "(7) FIAP - Grupo 69 - Tech Challenge - Fase 1 - YouTube")


# Relatórios PDF

- [Relatório principal](fiap_tech_challenge_fase_1_analise_cancer_csv_grupo69.pdf)
- [Relatório extra - visão computacional](fiap_tech_challenge_fase_1_extra_grupo69.pdf)


# Requisitos
- Python 3.9+ (para rodar local)
- Docker e Docker Compose (para rodar o jupyter notebook para treino do modelo e também a aplicação que consome os modelos produtizados em containers)

# Treinando os Modelos

## Treinamento do Modelo tabular

Dataset utilizados: 
https://www.kaggle.com/datasets/akashnath29/lung-cancer-dataset/data

O modelo pode ser treinado localmente utilizando o notebook `train_model/analise_cancer.ipynb`.

Alternativamente, você pode abrir este notebook diretamente no [Google Colab](https://colab.research.google.com/) para execução na nuvem, sem necessidade de configuração local.

Se estiver usando o ambiente Docker em `train_model/Dockerfile`, o container preserva o diretório `train_model`, expõe `datasets` na raiz para manter os paths dos notebooks e já inclui `pandoc`, `xelatex`, `texlive-fonts-recommended`, `texlive-latex-extra` e `texlive-plain-generic` para suportar a exportação de notebooks para PDF pelo Jupyter/nbconvert.

## Treinamento do Modelo de Imagem

Para o modelo de visão computacional, utilize o notebook `train_model/analise_cancer_imagem.ipynb`.

Este notebook realiza o treinamento de um classificador de imagens com `TensorFlow/Keras` a partir do dataset [IQ-OTH/NCCD](https://data.mendeley.com/datasets/bhmdr45bh2/4), incluindo preparação dos dados, aumento de imagens, ajuste de limiar e exportação do melhor modelo.


### Treinamento do model de visão computacional no Mac com processador silicon
Para treinar o modelo de visão computacional no Mac com processador silicon (M1, M2, M3, M4, etc) recomenda-se usar o conda e instalar o tensorflow-macos e tensorflow-metal habilitando assim o uso de gpu.
```bash
brew install --cask miniforge 
conda init zsh # ou bash se não estiver usando zsh
conda create -p ./.fiap-tf python=3.9
conda activate ./.fiap-tf
python -m pip install --upgrade pip setuptools wheel
pip install tensorflow-macos tensorflow-metal
# Não rode pip install tensorflow (isso instala a versão CPU/Intel e pode conflitar).
pip install -r req_no_tf_mac.txt # da o install no requirements sem instalar o tensorflow para poder usar o tensorflow-metal ja instalado
# se der erro no opencv-python tente conda install -c conda-forge opencv

#Verificação rápida — confirmar TF + Metal/GPU, se não der nenhum erro o setup no mac esta correto
python - <<'PY'
import time, numpy as np, tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Physical devices:", tf.config.list_physical_devices())
gpus = tf.config.list_physical_devices('GPU')
print("GPUs found:", gpus)
# Quick matmul to exercise device
a = tf.random.uniform([2000,2000])
b = tf.random.uniform([2000,2000])
t0 = time.time()
c = tf.matmul(a,b)
# force execution
_ = c.numpy()
print("Matmul time (s):", time.time() - t0)
PY
```


## Treinando com Docker
Para treinar usando o ambiente containerizado, utilize o Dockerfile em `train_model/Dockerfile`.

1. Construa a imagem a partir da raiz do projeto:
   ```bash
   docker build -f train_model/Dockerfile -t tech-challenge-train-model .
   ```
2. Execute o container expondo o Jupyter Lab:
   ```bash
   docker run --rm -p 8089:8888 tech-challenge-train-model
   ```
3. Acesse o Jupyter Lab em `http://localhost:8089` e abra o notebook `train_model/analise_cancer.ipynb` ou `train_model/analise_cancer_imagem.ipynb`.

Observação: o build precisa usar o contexto da raiz do repositório, porque o Dockerfile copia `requirements.txt` e a pasta `train_model/` a partir desse nível.


# Como rodar o projeto de exemplo usando o modelo produtizado

## Configurando variáveis de ambiente
A interpretação dos resultados via LLM (Gemini) precisa de uma chave de API. Copie o arquivo de exemplo e preencha `GEMINI_API_KEY` com sua chave:
```bash
cp .env.example .env
```
Esse `.env` é lido tanto na execução local (via `python-dotenv`) quanto pelo `docker-compose` (o serviço `tech-challenge-backend` carrega o arquivo através de `env_file`).

## Rodando localmente (sem Docker)
1. Crie e ative um ambiente virtual com Python 3.10+:
	```bash
	python3 -m venv venv
	source venv/bin/activate  # Linux/macOS
	venv\Scripts\activate   # Windows
	```
2. Instale as dependências:
	```bash
	# no linux/windows
	pip install -r requirements.txt

	# no mac
	pip install -r req_no_tf_mac.txt # da o install no requirements sem instalar o tensorflow para poder usar o tensorflow-metal ja instalado
	```
3. Inicie a API:
	```bash
	uvicorn backend.main:app --host 0.0.0.0 --port 8888 --reload
	```
4. Inicie a interface web:
	```bash
	streamlit run frontend/app.py
	```

## Rodando com Docker e Docker Compose
1. Certifique-se de que Docker e Docker Compose estejam instalados.
2. Construa e suba os serviços:
	```bash
	docker-compose up
	```
3. A API estará disponível em `http://localhost:8888`.
4. A aplicação web estará disponível em `http://localhost:8501`.

# Notas
- A interface web faz primeiro a pré-triagem por CSV e depois libera a segunda etapa para envio das imagens de ressonância.
- Utilize o campo de upload da web para carregar múltiplas imagens de exames.
- Caso queira limpar resultados, use o botão de limpar.
