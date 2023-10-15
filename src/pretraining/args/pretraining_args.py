# coding=utf-8
# Copyright 2021 Intel Corporation. All rights reserved.
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

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PretrainScriptParamsArguments:
    """
    PretrainScriptParamsArguments
    """

    _argument_group_name = "Pretraining Arguments"

    seed: int = field(default=42, metadata={"help": "random seed for initialization)"})

    output_dir: str = field(
        default=None,
        metadata={"help": "The output directory where the model checkpoints will be written."},
    )

    max_predictions_per_seq: int = field(
        default=20,
        metadata={"help": "The maximum number of masked tokens in a sequence to be predicted."},
    )

    local_rank: int = field(
        default=-1, metadata={"help": "local_rank for distributed training on gpus)"}
    )

    load_training_checkpoint: str = field(
        default=None,
        metadata={
            "help": "This is the path to the TAR file which contains model+opt state_dict() checkpointed."
        },
    )

    load_checkpoint_id: str = field(
        default=None, metadata={"help": "Checkpoint identifier to load from checkpoint path"}
    )

    num_epochs_between_checkpoints: int = field(
        default=-1,
        metadata={"help": "Number of epochs between a full checkpoint (used for pre-training)"},
    )

    job_name: str = field(
        default="pretraining_experiment",
        metadata={"help": "Experiment job name"},
    )

    project_name: str = field(
        default="budget-lm-pretraining", metadata={"help": "Project name (W&B)"}
    )

    max_steps: int = field(
        default=9223372036854775807,
        metadata={"help": "Maximum number of training steps of effective batch size to complete."},
    )

    max_steps_per_epoch: int = field(
        default=9223372036854775807,
        metadata={
            "help": "Maximum number of training steps of effective batch size within an epoch to complete."
        },
    )

    print_steps: int = field(
        default=100, metadata={"help": "Interval to print training details.)"}
    )

    do_validation: bool = field(
        default=False, metadata={"help": "Enable/Disable running validation"}
    )

    validation_epochs: int = field(
        default=10, metadata={"help": "Number of epochs between running validation evaluation"}
    )

    validation_epochs_begin: int = field(
        default=1,
        metadata={"help": "Number of epochs between running validation evaluation in first stage"},
    )

    validation_epochs_end: int = field(
        default=1,
        metadata={"help": "Number of epochs between running validation evaluation in last stage"},
    )

    validation_begin_proportion: float = field(
        default=0.1, metadata={"help": "how long does the first stage of training last?"}
    )

    validation_end_proportion: float = field(
        default=0.1, metadata={"help": "how long does the last stage of training last?"}
    )

    validation_micro_batch: int = field(
        default=16, metadata={"help": "Per device validation batch size"}
    )

    add_nsp: bool = field(
        default=False,
        metadata={"help": "Create a model with NSP task; dataset assumed to have NSP labels"},
    )

    current_run_id: str = field(
        default="", metadata={"help": "the current run id (mainly for hyperparams)"}
    )

    early_exit_time_marker: float = field(
        default=24.0, metadata={"help": "Max hours for pre-training)"}
    )

    total_training_time: float = field(
        default=24.0, metadata={"help": "Max hours for pre-training)"}
    )

    finetune_time_markers: str = field(
        default=None,
        metadata={"help": "Time markers for saving fine-tuning checkpoints"},
    )

    finetune_checkpoint_at_end: bool = field(
        default=True,
        metadata={"help": "Save a finetuning checkpoint when training is done."},
    )

    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "gradient accumulation steps"}
    )

    train_batch_size: int = field(default=65536, metadata={"help": "train_batch_size"})

    train_micro_batch_size_per_gpu: int = field(
        default=32, metadata={"help": "train_micro_batch_size_per_gpu"}
    )
    num_epochs: int = field(
        default=1000000, metadata={"help": "Number of training epochs"}
    )

    lr: float = field(default=0.011, metadata={"help": "lr"})

    use_early_stopping: bool = field(
        default=False,
        metadata={"help": "Should use early stopping?"},
    )

    early_stop_time: int = field(
        default=720,
        metadata={"help": "The point after which the run should perform well enough? (in MINUTES)"},
    )

    early_stop_eval_loss: float = field(
        default=2.1,
        metadata={
            "help": "The upper bound value of the validation loss when *early_stop_time* has reached?"
        },
    )
        
    scale_cnt_limit: int = field(
        default=100,
        metadata={"help": "The limit of the number of times the scale in the optimizer reached 1, in order to early stop."},
    )
    
    log_throughput_every: int = field(
        default=20,
        metadata={"help": "How many steps should the throughput (im Samples/s) be logged."},
    )

    def __post_init__(self):
        self.no_nsp = not self.add_nsp
        self.learning_rate = self.lr
        if self.finetune_time_markers is not None:
            self.finetune_time_markers = [
                float(x.strip()) for x in self.finetune_time_markers.split(",")
            ]
