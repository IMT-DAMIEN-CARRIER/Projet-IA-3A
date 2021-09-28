# syntax=docker/dockerfile:1
FROM bitnami/pytorch:latest

WORKDIR /opt/app/

COPY *.py ./

COPY . .

EXPOSE 1111

CMD ["pip", "install", "torchvision"]