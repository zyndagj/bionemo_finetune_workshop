# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import tempfile
from pathlib import Path
from typing import Sequence, Tuple

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, RichModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm as nllm
from nemo.lightning import resume
from nemo.lightning.nemo_logger import NeMoLogger
from nemo.lightning.pytorch import callbacks as nl_callbacks
from nemo.lightning.pytorch.callbacks.model_transform import ModelTransform
from nemo.lightning.pytorch.callbacks.peft import PEFT
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule

from bionemo.core.data.load import load
from bionemo.esm2.api import ESM2GenericConfig
from bionemo.esm2.data.tokenizer import BioNeMoESMTokenizer, get_tokenizer
from bionemo.esm2.model.finetune.datamodule import ESM2FineTuneDataModule
from bionemo.esm2.model.finetune.finetune_regressor import ESM2FineTuneSeqConfig, InMemorySingleValueDataset
from bionemo.llm.model.biobert.lightning import biobert_lightning_module


__all__: Sequence[str] = ("train_model",)


def train_model(
    experiment_name: str,
    experiment_dir: Path,
    config: ESM2GenericConfig,
    data_module: pl.LightningDataModule,
    n_steps_train: int,
    metric_tracker: Callback | None = None,
    tokenizer: BioNeMoESMTokenizer = get_tokenizer(),
    peft: PEFT | None = None,
    _use_rich_model_summary: bool = True,
) -> Tuple[Path, Callback | None, nl.Trainer]:
    """Trains a BioNeMo ESM2 model using PyTorch Lightning.

    Parameters:
        experiment_name: The name of the experiment.
        experiment_dir: The directory where the experiment will be saved.
        config: The configuration for the ESM2 model.
        data_module: The data module for training and validation.
        n_steps_train: The number of training steps.
        metric_tracker: Optional callback to track metrics
        tokenizer: The tokenizer to use. Defaults to `get_tokenizer()`.
        peft: The PEFT (Parameter-Efficient Fine-Tuning) module. Defaults to None.
        _use_rich_model_summary: Whether to use the RichModelSummary callback, omitted in our test suite until
            https://nvbugspro.nvidia.com/bug/4959776 is resolved. Defaults to True.

    Returns:
        A tuple containing the path to the saved checkpoint, a MetricTracker
        object, and the PyTorch Lightning Trainer object.
    """
    checkpoint_callback = nl_callbacks.ModelCheckpoint(
        save_last=True,
        save_on_train_epoch_end=True,
        monitor="reduced_train_loss",  # TODO find out how to get val_loss logged and use "val_loss",
        every_n_train_steps=n_steps_train // 2,
        always_save_context=True,  # Enables the .nemo file-like checkpointing where all IOMixins are under SerDe
    )

    # Setup the logger and train the model
    nemo_logger = NeMoLogger(
        log_dir=str(experiment_dir),
        name=experiment_name,
        tensorboard=TensorBoardLogger(save_dir=experiment_dir, name=experiment_name),
        ckpt=checkpoint_callback,
    )
    # Needed so that the trainer can find an output directory for the profiler
    # ckpt_path needs to be a string for SerDe
    optimizer = MegatronOptimizerModule(
        config=OptimizerConfig(
            lr=5e-4,
            optimizer="adam",
            use_distributed_optimizer=True,
            fp16=config.fp16,
            bf16=config.bf16,
        )
    )
    module = biobert_lightning_module(config=config, tokenizer=tokenizer, optimizer=optimizer, model_transform=peft)

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        ddp="megatron",
        find_unused_parameters=True,
        enable_nemo_ckpt_io=True,
    )

    if _use_rich_model_summary:
        # RichModelSummary is not used in the test suite until https://nvbugspro.nvidia.com/bug/4959776 is resolved due
        # to errors with serialization / deserialization.
        callbacks: list[Callback] = [RichModelSummary(max_depth=4)]
    else:
        callbacks = []

    if metric_tracker is not None:
        callbacks.append(metric_tracker)
    if peft is not None:
        callbacks.append(
            ModelTransform()
        )  # Callback needed for PEFT fine-tuning using NeMo2, i.e. biobert_lightning_module(model_transform=peft).

    trainer = nl.Trainer(
        accelerator="gpu",
        devices=1,
        strategy=strategy,
        limit_val_batches=2,
        val_check_interval=n_steps_train // 4,
        max_steps=n_steps_train,
        num_nodes=1,
        log_every_n_steps=n_steps_train // 4,
        callbacks=callbacks,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
    )
    nllm.train(
        model=module,
        data=data_module,
        trainer=trainer,
        log=nemo_logger,
        resume=resume.AutoResume(
            resume_if_exists=True,  # Looks for the -last checkpoint to continue training.
            resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
        ),
    )
    ckpt_path = Path(checkpoint_callback.last_model_path.replace(".ckpt", ""))
    return ckpt_path, metric_tracker, trainer


import requests, pandas, os
from io import BytesIO
import pandas
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # set the results directory
    experiment_results_dir = tempfile.TemporaryDirectory().name

    # create a List[Tuple] with (sequence, target) values
    #model_checkpoint = "facebook/esm2_t12_35M_UR50D"
    model_checkpoint = "facebook/esm2_t33_650M_UR50D"
        
    query_url = "https://rest.uniprot.org/uniprotkb/stream?compressed=true&fields=accession%2Csequence%2Ccc_subcellular_location&format=tsv&query=%28%28organism_id%3A9606%29%20AND%20%28reviewed%3Atrue%29%20AND%20%28length%3A%5B80%20TO%20500%5D%29%29"
    tmp_file = "/tmp/uniprot.parquet.gz"
    
    if not os.path.exists(tmp_file):
        # Get data
        uniprot_request = requests.get(query_url)
        # Store data as binary object
        bio = BytesIO(uniprot_request.content)
        # Reach bionary object as compressed csv
        df = pandas.read_csv(bio, compression='gzip', sep='\t')
        # Write out to local location for faster reloads
        df.to_parquet(tmp_file, compression="gzip")
    else:
        # Load from checkpoint
        df = pandas.read_parquet(tmp_file)

    df = df.dropna()  # Drop proteins with missing columns
    cytosolic = df['Subcellular location [CC]'].str.contains("Cytosol") | df['Subcellular location [CC]'].str.contains("Cytoplasm")
    membrane = df['Subcellular location [CC]'].str.contains("Membrane") | df['Subcellular location [CC]'].str.contains("Cell membrane")
    cytosolic_df = df[cytosolic & ~membrane]
    membrane_df = df[membrane & ~cytosolic]
    cytosolic_sequences = cytosolic_df["Sequence"].tolist()
    cytosolic_labels = [0 for protein in cytosolic_sequences]
    membrane_sequences = membrane_df["Sequence"].tolist()
    membrane_labels = [1 for protein in membrane_sequences]

    sequences = cytosolic_sequences + membrane_sequences
    labels = cytosolic_labels + membrane_labels

    # Quick check to make sure we got it right
    len(sequences) == len(labels)

    train_sequences, test_sequences, train_labels, test_labels = train_test_split(sequences, labels, test_size=0.25, shuffle=True)
    train_dataset = InMemorySingleValueDataset(list(zip(train_sequences, train_labels)))
    test_dataset = InMemorySingleValueDataset(list(zip(test_sequences, test_labels)))

    data_module = ESM2FineTuneDataModule(train_dataset=train_dataset, valid_dataset=test_dataset, micro_batch_size=8)


    experiment_name = "finetune_regressor"
    n_steps_train = 200
    seed = 42

    # To download a 650M pre-trained ESM2 model
    pretrain_ckpt_path = load("esm2/650m:2.0")

    config = ESM2FineTuneSeqConfig(initial_ckpt_path=str(pretrain_ckpt_path))

    checkpoint, metrics, trainer = train_model(
        experiment_name=experiment_name,
        experiment_dir=Path(experiment_results_dir),  # new checkpoint will land in a subdir of this
        config=config,  # same config as before since we are just continuing training
        data_module=data_module,
        n_steps_train=n_steps_train,
    )
    print(f"Experiment completed with checkpoint stored at {checkpoint}")
