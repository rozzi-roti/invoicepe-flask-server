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

CMD exec gunicorn --bind :5000 --workers 1 --threads 8 --timeout 0 main:app