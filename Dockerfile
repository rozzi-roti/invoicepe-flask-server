# syntax=docker/dockerfile:1

FROM python:3.11-slim

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

WORKDIR /python-docker

COPY requirements.txt requirements.txt

RUN apt-get update

RUN apt-get -y install gcc

RUN apt-get update -qqy && apt-get install -qqy \
        tesseract-ocr \
        libtesseract-dev

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY . .

CMD exec unicorn --config config.py -w 8 --worker-class gevent --preload -b 0.0.0.0:5000 app:app