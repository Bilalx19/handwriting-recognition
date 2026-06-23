# This is the Dockerfile for the Client
# For now I will create one Dockerfile for every client -> this will be changed to one Dockerfile (container) for each client

# To build do: docker build -f client.Dockerfile -t handwriting-client .
# To run only the Client (one Container with all the clients) do: docker run --rm handwriting-client
# Running the Container wont return a result (just empty) since it needs the server to be running and the client to be connected to the server
# To check the whats inside the container do: docker run --rm -it handwriting-client ls -la

FROM python:3.12-slim

WORKDIR /app

COPY requirements_client.txt .

RUN pip install --no-cache-dir -r requirements_client.txt

COPY FL/fl_client.py ./FL/

COPY models/cnn.py ./models/

COPY data.py .

COPY train.py .

ENV PYTHONPATH=/app

ENV PYTHONBUFFERED=1

# to connect to the server we set the ENV
ENV SERVER_ADDRESS=server:8080

CMD ["python", "-m", "FL.fl_client"]