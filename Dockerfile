## Here the Images are built


## (FROM) Defines a new build stage and sets the base image for that stage (Source 2)
## We use python 3.12 since the code runs on this version without any issues
FROM python:3.12

## WORKDIR	Sets the working directory for the commands that follow it (Source 2)
WORKDIR /app

## since docker creates layers it is better to Copy the requirements.txt first so if any changes are made in Requirements.txt, it wont copy the whole other files again???
COPY requirements.txt .


## (RUN) Executes any commands it is given in a new layer on top of the current image that has been built up to that point (Source 2)
## Here we can add our dependencies and install them in the image, but we can also use a requirements.txt (Source 1)
## having the dependencies in a txt file is probably cleaner
RUN pip install --no-cache-dir -r requirements.txt
# --no-cache-dir is used to prevent pip from caching the installed packages
# this reduces the disk space needed
# -r reads the .txt file and installs all the packages in it, without it we would only install the first dependency listed in the file

## COPY	Copies the contents from a source directory to the filesystem at the path passed in to a new layer in the image (Source 2)
COPY . .

## ADD	A more advanced version of COPY that supports things like local tar extraction and remote URLs (Source 2)
# ADD # not needed?

## ENTRYPOINT	Configures the executables or commands that will run once the container is initialized (Source 2)
# ENTRYPOINT # not needed?

## USER	Sets the user that the container is run under, often used to run containers as non-root (Source 2)
# USER # not needed?

## LABEL	Adds key-value labels to the image being built; note that labels are passed down from base images (Source 2)
# LABEL # not needed?

## ARG	Define build-time only variables that can be used during the Docker image build (Source 2)
# ARG # not needed?

## ENV	Sets environment variables from within the Docker image that can be used during the build process or when the container is run (Source 2)
ENV PYTHONBUFFERED=1
# so all prints/output happen in real time and not after the program finishes
# I added it for debugging reasons

## EXPOSE	Defines a port that the container will listen on when the image is run as a container (Source 2)
# EXPOSE # not needed?
EXPOSE 8501
# but why EXPOSE 8501??? it was suggested by AI

## VOLUME	Creates a mount point with a specific name that is bound to a mounted volume from the underlying host or another container (Source 2)
# VOLUME # not needed?

## CMD	Defines the default set of arguments that are supplied to the process that runs the container when it's launched via ENTRPOINT (Source 2)
#CMD ["python", "-m", "streamlit", "run", "UI/app.py"]
CMD ["python", "-m", "streamlit", "run", "UI/app.py", "--server.address=0.0.0.0"]  
## but why --server.address=0.0.0.0 ??? it was suggested by AI
## runs: python -m streamlit run UI/app.py