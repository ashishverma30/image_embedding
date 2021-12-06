FROM python:3.8


# install build utilities
RUN apt-get update && \
	apt-get install -y gcc make apt-transport-https ca-certificates build-essential

# python environment
RUN python3 --version
RUN pip3 --version

# set the working directory 
WORKDIR  /usr/src/<app-name>

# Installing python dependencies and libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy files from the root directory to the working directory
COPY src/ /src/
RUN ls -la /src/*

# Running Python Application
CMD ["python3", "/src/docker_test.py"]