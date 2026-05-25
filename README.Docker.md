### What you need to install to run Docker

1. Go to this website and install Docker for youre Operating System:
https://www.docker.com/products/docker-desktop/

2. Afterwards check if its installed properly trough this command:
docker --version

3. Optional and recommendet: Install the Docker and Docker compose extentions in youre IDE


### Goal

We will Containerize this whole Project into different Containers. Each Client will be in a different Container (recommended trough our Prof. while last weekly meeting). Furthermore we will Containerize the Data processing, the models, and the app/UI.

==> But we will first create one big whole Container to test Docker. In the next steps we will create multiple containers.
And then we will use Docker compose and create a "multi-container Application" (Source 1)

### Usefull Sources

In the following the sources are listed which were used while the containerization process and can be helpful to understand the Docker Part:


- https://www.datacamp.com/tutorial/docker-tutorial (Source 1)
- https://depot.dev/blog/docker-build-image#what-is-docker (Source 2)

- https://youtu.be/DQdB7wFEygo?si=QYfeu4HelSzoWWNO (brainrot explanation)


### Basic knowledge:

Docker Images (Recipe) -> Contains the fundamental building blocks of containers. They are built in the Dockerfile. 

## Building a container:
DO:
docker build -t handwriting-app .
this builds a container (we only need one first for testing)
this takes apperantly few minutes

## RUN
RUN it trough command: 
docker run -p 8501:8501 handwriting-app

### Building and running your application

When you're ready, start your application by running:
`docker compose up --build`.

### Deploying your application to the cloud

First, build your image, e.g.: `docker build -t myapp .`.
If your cloud uses a different CPU architecture than your development
machine (e.g., you are on a Mac M1 and your cloud provider is amd64),
you'll want to build the image for that platform, e.g.:
`docker build --platform=linux/amd64 -t myapp .`.

Then, push it to your registry, e.g. `docker push myregistry.com/myapp`.

Consult Docker's [getting started](https://docs.docker.com/go/get-started-sharing/)
docs for more detail on building and pushing.