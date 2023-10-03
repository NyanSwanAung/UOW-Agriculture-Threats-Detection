FROM python:3.8

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8888

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install other LINUX dependencies
RUN apt install wget
RUN apt --fix-broken -y install
RUN apt update
RUN apt-get -y install xvfb
RUN apt-get install nano
RUN apt-get install zip
RUN apt-get install tree

# if there is no app dir, it creates app dir
# if there is app dir, it goes into app dir
WORKDIR /app

# Copy all the files from current dir to app
COPY . /app

# Install requirements and other libraries
RUN pip install -U pip && pip install -r requirements.txt
RUN pip install jupyter -U && pip install jupyterlab

# Change working dir to app
WORKDIR /app

CMD ["python3", "main.py"]

#ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root"]