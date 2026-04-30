FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY web/app.py ./app.py
COPY web/static/logo.png ./static/logo.png

EXPOSE 8501
CMD ["streamlit", "run", "/app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
