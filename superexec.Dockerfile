#Docker Image welches ggf. in die Cloud gepusht werden kann.

FROM flwr/superexec:1.31.0

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY data.py train.py testing.py ./
COPY FL ./FL
COPY models ./models
COPY flower-config.toml /root/.flwr/config.toml

RUN sed -i '/flwr\[simulation\]/d' pyproject.toml \
    && python -m pip install -U --no-cache-dir .

ENTRYPOINT ["flower-superexec"]