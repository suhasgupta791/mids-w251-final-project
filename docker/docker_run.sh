#!/bin/bash -f 

docker run -d \
	--runtime nvidia \
	-v $(pwd):/root \
	-v /tmp:/tmp \
	-p 8888:8888 \
	-p 8000:8000 \
	-p 6006:6006 \
	--name=w251-project \
	pytorch_cuda
