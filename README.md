# bionemo_finetune_workshop
Fine-tuning ESM2 with BioNeMo

## Requirements

- Container runtime
- Ability to connect to Jupyter instance from compute node

## 1. Launching the container

### Docker

```
docker pull nvcr.io/nvidia/clara/bionemo-framework:2.3
docker run --rm -it \
    -v ${PWD}/scripts:/home/${USER} -e HOME=/home/${USER} \
    -v /etc/passwd:/etc/passwd:ro \
    -u $(shell id -u):$(shell id -g) -e USER=$(USER) \
    -p 8888:8888 --gpus 1 --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864
    nvcr.io/nvidia/clara/bionemo-framework:2.3 bash -l
```

> You may need to move data to /tmp if /home is a shared filesystem

Alternatively, you can run

```
make pull
make run
```

### Enroot

```
enroot import docker://nvcr.io#nvidia/clara/bionemo-framework:2.3
enroot 
enroot start blah
```

### Singularity/Apptainer

> Please substitute singularity/apptainer based on what you use

```
apptainer pull docker://nvcr.io/nvidia/clara/bionemo-framework:2.3
apptainer exec --nv
```

## 2. Starting Jupyter

Once inside the container, start Jupyter Lab using

```
jupyter-lab 
```

and then navigate to your HOME directory.

## Next-steps

- Pre-training
- Explore other models with BioNeMo
- Explore model parallelism
- Explore multi-node training runs
- Contact us about porting your model to BioNeMo