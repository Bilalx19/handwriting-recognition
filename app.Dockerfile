# This is the Dockerfile for the app (streamlit)

# To build do: docker build -f app.Dockerfile -t handwriting-app .
# To run only the app (one Container with all the clients) do: docker run --rm handwriting-app

# Running the Container wont return a result (just empty) since it needs the server to be running and the client to be connected to the server
# To check the whats inside the container do: docker run --rm -it handwriting-app ls -la

FROM python:3.12-slim 

WORKDIR /app

#COPY requirements_app.txt .
COPY requirements_app.txt .
# When using COPY with more than one source file, the destination must be a directory and end with a / or a

#RUN pip install --no-cache-dir -r requirements_app.txt
RUN pip install --no-cache-dir -r requirements_app.txt

COPY UI/ ./UI/

COPY models/ ./models/

COPY final_model.pt ./final_model.pt

ENV PYTHONBUFFERED=1

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "UI/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
