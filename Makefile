TAG := 2.3
#TAG := 2.2
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

scripts:
	mkdir scripts

/tmp/$(USER)/scripts: | scripts
	rsync -ra $| $(dir $@)

/tmp/$(USER)/bionemo-framework:
	[ -d $(dir $@) ] || mkdir -p $(dir $@)
	cd $(dir $@)
	git clone -b v$(TAG) $(REPO)

/tmp/$(USER)/bionemo-framework/sub-packages/bionemo-dnabert2: bionemo-dnabert2 | /tmp/$(USER)/bionemo-framework
	cp -r $< $@

/tmp/$(USER)/bionemo-framework/sub-packages/DNABERT-2-117M: | /tmp/$(USER)/bionemo-framework
	git clone https://huggingface.co/zhihan1996/DNABERT-2-117M $@

#targets := $(shell echo /tmp/$(USER)/bionemo-framework/sub-packages/{bionemo-dnabert2,DNABERT-2-117M} /tmp/$(USER)/scripts)
targets := /tmp/$(USER)/scripts
run: | $(targets)
	docker run --rm -it $(GPUS) $(CURRENT_USER) $(IMG)
	rsync -ra /tmp/$(USER)/scripts ./
