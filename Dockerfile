# syntax=docker/dockerfile:1
FROM bitnami/pytorch:latest

WORKDIR /opt/app/

RUN pip install --no-cache-dir torchvision matplotlib numpy