#!/bin/bash -f 

docker run -d \
	--runtime nvidia \
	-v $(pwd):/root \
	-v /tmp:/tmp \
	-p 8888:8888 \
	-p 8000:8000 \
	-p 6007:6006 \
	-p 8080:8080 \
	--name=w251-project \
	-t pytorch_cuda \
       	/bin/bash
