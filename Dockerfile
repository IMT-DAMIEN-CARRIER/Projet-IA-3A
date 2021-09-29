# syntax=docker/dockerfile:1
FROM bitnami/pytorch:latest

WORKDIR /opt/app/

COPY . .

EXPOSE 1111

RUN pip install --no-cache-dir torchvision