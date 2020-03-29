#!/bin/bash -f 

docker run -d \
	--runtime nvidia \
	-v $(pwd):/root \
	-v /tmp:/tmp \
	-p 8887:8888 \
	-p 8002:8000 \
	-p 6007:6006 \
	--name=w251-project \
	pytorch_cuda
