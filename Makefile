IMAGE_NAME = "image-projet-ia"
CONTAINER_NAME = "container-projet-ia"
CURRENT_DIR = $(shell pwd)

up: build run

build:
	docker build . -t ${IMAGE_NAME}

run:
	docker run --name ${CONTAINER_NAME} -v ${CURRENT_DIR}:/opt/app -it ${IMAGE_NAME}:latest bash

clear: down remove-container remove-image

down:
	docker stop ${CONTAINER_NAME}

remove-image:
	docker image rm ${IMAGE_NAME}:latest

remove-container:
	docker rm ${CONTAINER_NAME}
