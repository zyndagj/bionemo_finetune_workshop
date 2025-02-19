TAG := 2.3
IMG := nvcr.io/nvidia/clara/bionemo-framework:$(TAG)
REPO := https://github.com/NVIDIA/bionemo-framework.git
CURRENT_USER := -v /tmp/$(USER):/home/$(USER) -e HOME=/home/$(USER) -v /etc/passwd:/etc/passwd:ro -u $(shell id -u):$(shell id -g) -v /etc/group:/etc/group:ro -v /etc/shadow:/etc/shadow:ro -e USER=$(USER) -p 8888:8888 -p 6006:6006
ifdef NV_GPU
	GPUS := --gpus \"device=$${NV_GPU}\" --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
else
	GPUS := --gpus 1 --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
endif
SHELL = /bin/bash
.ONESHELL:

pull:
	docker pull $(IMG)

/tmp/$(USER)/scripts: | scripts
	rsync -ra $| $(dir $@)

run: | /tmp/$(USER)/scripts
	docker run --rm -it $(GPUS) $(CURRENT_USER) $(IMG)
	rsync -ra /tmp/$(USER)/scripts ./
