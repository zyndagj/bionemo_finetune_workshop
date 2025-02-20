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
enroot import --output bionemo.sqsh docker://nvcr.io#nvidia/clara/bionemo-framework:2.3
enroot create --name bionemo bionemo.sqsh
enroot start bionemo
```

Alternatively, you can run

```
make enroot
```

### Singularity/Apptainer

> Please substitute singularity/apptainer based on what you use

```
apptainer pull docker://nvcr.io/nvidia/clara/bionemo-framework:2.3
apptainer exec --nv <IMG> jupyter-lab --notebook-dir=$PWD
```

If you're on Bruno, you can also use my cached container at

```
make bruno
```

## 2. Starting Jupyter

Once inside the container, navigate to the directory containing this repo. In Docker, that may be `$HOME`, for enroot and apptainer, that will be the original path. After that, start Jupyter Lab using

```
jupyter-lab --notebook-dir=.
```

## Next-steps

- Pre-training
- Explore other models with BioNeMo
- Explore model parallelism
- Explore multi-node training runs
- Contact us about porting your model to BioNeMo
