#!/bin/bash -f 

docker run -d \
	--runtime nvidia \
	-v /mnt/sdb1/w251/mids-w251-final-project:/root \
	-v /tmp:/tmp \
	-p 8888:8888 \
	-p 8000:8000 \
	--name=w251-project \
	pytorch_cuda