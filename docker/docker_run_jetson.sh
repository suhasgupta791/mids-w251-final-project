#!/bin/bash -f 

docker run -d \
	--runtime nvidia \
	-v $(pwd):/root \
	-v /tmp:/tmp \
	-p 8889:8888 \
	-p 8001:8000 \
	-p 6008:6006 \
	--gpus all \
	--privileged \
	--name=w251-project \
	inference_tf_pytorch
	#tensorrt_project
