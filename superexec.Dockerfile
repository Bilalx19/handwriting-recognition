FROM flwr/superexec:1.32.1

WORKDIR /app

# Strip flwr[simulation] (Flower comes from the base image; Ray/simulation is not used for
# deployment) and install the remaining runtime dependencies. App code is delivered as a FAB.
COPY pyproject.toml LICENSE ./
RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml \
    && python -m pip install -U --no-cache-dir .

ENTRYPOINT ["flower-superexec"]
