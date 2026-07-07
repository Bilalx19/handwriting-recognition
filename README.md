# handwriting-recognition

Federated handwriting recognition on the FEMNIST dataset (62 character classes). Four clients, each standing in for a different school, train a shared CNN with Flower without exchanging raw handwriting samples; only model updates are aggregated. Global model accuracy is about 86% on client validation data.

## Stack

PyTorch (CNN), Flower (federated learning), Docker, Google Kubernetes Engine.

## Layout

- `data.py`, `train.py`, `models/`: dataset loading, training loop, and CNN.
- `FL/`: Flower ServerApp (FedAvg with per client contribution tracking) and ClientApp.
- `dashboard/`, `UI/`: Streamlit dashboards and a handwriting input app.
- `GCP/`: Kubernetes manifests for the GKE deployment.

## Run the full system locally (Docker)

Starts the SuperLink, server, four clients, and dashboards:

    docker compose up --build

The dashboard is served on http://localhost:8501.

## Run the inference app only (without Docker)

    pip install torch torchvision flwr flwr-datasets[vision] numpy pillow streamlit streamlit-drawable-canvas plotly pandas
    python -m streamlit run UI/app.py

## Deploy to the cloud (GKE)

The `GCP/` manifests run the same system on GKE using Flower's process isolation setup. To reproduce it you need the Google Cloud CLI (with `kubectl` and the GKE auth plugin), Docker, and a Google Cloud project with billing enabled.

Set your project and enable the required services:

    PROJECT_ID=your-project-id
    gcloud config set project "$PROJECT_ID"
    gcloud services enable artifactregistry.googleapis.com container.googleapis.com \
      cloudbuild.googleapis.com compute.googleapis.com

Create the image repository:

    gcloud artifacts repositories create flower-gcp-example-artifacts \
      --repository-format=docker --location=us-central1

Build and push the application image:

    gcloud builds submit --config cloudbuild.yaml .

Create the cluster and connect to it:

    gcloud container clusters create flower-cluster \
      --zone us-central1-a --num-nodes 2 --machine-type e2-standard-4
    gcloud container clusters get-credentials flower-cluster --zone us-central1-a

Point the manifests at your project, deploy, and wait until every pod reports Running:

    sed -i "s/fml-handwriting-2026/$PROJECT_ID/g" GCP/*.yaml
    kubectl apply -f GCP/
    kubectl get pods

Read the SuperLink external IP and set it as the `gcp` address in `flower-config.toml`:

    kubectl get svc superlink-service

Submit the training run:

    flwr run . gcp --stream

Save the model and delete the cluster:

    POD=$(kubectl get pods -l app=superexec-serverapp -o jsonpath='{.items[0].metadata.name}')
    kubectl cp "$POD":/app/model/final_model.pt ./final_model.pt
    gcloud container clusters delete flower-cluster --zone us-central1-a
