# Creates a layer from the python:3.8 Docker image
FROM python:3.8

WORKDIR /fremoji
# Copy all the files from the folders the Dockerfile is to the container root folder
COPY requirements.txt .
# Install the modules specified in the requirements.txt
RUN pip3 install -r requirements.txt

COPY . . 
# The port on which a container listens for connection
EXPOSE 8501

# The command that run the app
CMD [ "streamlit", "run", "app.py"]