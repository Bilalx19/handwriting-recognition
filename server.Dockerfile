# This is the Dockerfile for the model server

# To build do: docker build -f server.Dockerfile -t handwriting-server .
# To run only the Server (one Container with all the clients) do: docker run --rm handwriting-server

# Running the Container wont return a result (just empty) since it needs the server to be running and the client to be connected to the server
# To check the whats inside the container do: docker run --rm -it handwriting-server ls -la

FROM python:3.12-slim

WORKDIR /app

#COPY requirements_server.txt .
COPY requirements_server.txt .

#RUN pip install --no-cache-dir -r requirements_server.txt
RUN pip install --no-cache-dir -r requirements_server.txt

COPY FL/fl_server.py ./FL/

COPY  final_model.pt ./final_model.pt
# server has the final model

COPY models/cnn.py ./models/

ENV PYTHONPATH=/app
ENV PYTHONBUFFERED=1

EXPOSE 8080

CMD ["python", "-m", "FL.fl_server"]



