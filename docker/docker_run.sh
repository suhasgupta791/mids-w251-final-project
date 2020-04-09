#!/bin/bash -f 

nohup docker run -d \
	--runtime nvidia \
	-v $(pwd):/root \
	-v /mnt/sdb1/w251/data_root:/data_root \
	-v /tmp:/tmp \
	-p 8888:8888 \
	-p 8000:8000 \
	-p 6007:6006 \
	-p 8080:8080 \
	--name=w251-project \
	-t pytorch_cuda \
       	/bin/bash
