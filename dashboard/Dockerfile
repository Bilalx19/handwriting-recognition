FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir streamlit plotly pandas

COPY app.py .
COPY client_app.py .

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
