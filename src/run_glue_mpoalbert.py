# coding=utf-8
# Copyright 2021 Intel Corporation. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import json
import logging
import os
import random
import sys
import glob
import torch
from argparse import Namespace
from pretraining.args.deepspeed_args import remove_cuda_compatibility_for_kernel_compilation
from pretraining.albert import AlbertForSequenceClassification
from transformers import AlbertForSequenceClassification as AlbertForSequenceClassificationOri
from transformers import AlbertConfig as AlbertConfigOri
from pretraining.configs import PretrainedBertConfig, PretrainedAlbertConfig
from pretraining.base import model_from_pretrained_mpo
from dataclasses import dataclass, field
from typing import Optional
import uuid

import numpy as np
import transformers
from datasets import load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from pretraining.compress_tools_v2.trainer_callback import EarlyStoppingCallback
from pretraining.compress_tools_v2.trainer_utils import SchedulerType
from pretraining.compress_tools_v2.custom_mpo_args import CustomArguments
from pretraining.sharer import Sharer, chain_module_names, chain_module_names_mpoalbert
from task_config import *

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
MODELS = {
    "albert-frominit": (AlbertForSequenceClassification, PretrainedAlbertConfig, AutoTokenizer),
    "albert-ori": (AlbertForSequenceClassificationOri, AlbertConfigOri, AutoTokenizer)
}
logger = logging.getLogger(__name__)

def get_parameter_number(net):
    '''
    :param net: model class
    :return: params statistics
    '''
    total_num = sum(p.numel() for p in net.parameters())/1000/1000
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)/1000/1000
    return {'Total(M)': total_num, 'Trainable(M)': trainable_num}
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError(
                    "Unknown task, you should pick one in " + ",".join(task_to_keys.keys())
                )
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )


@dataclass
class FinetuneTrainingArguments(TrainingArguments):
    group_name: Optional[str] = field(default=None, metadata={"help": "W&B group name"})
    project_name: Optional[str] = field(default=None, metadata={"help": "Project name (W&B)"})
    early_stopping_patience: Optional[int] = field(
        default=-1, metadata={"help": "Early stopping patience value (default=-1 (disable))"}
    )
    # overriding to be True, for consistency with final_eval_{metric_name}
    fp16_full_eval: bool = field(
        default=True,
        metadata={"help": "Whether to use full 16-bit precision evaluation instead of 32-bit"},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    unique_run_id = str(uuid.uuid1())

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, FinetuneTrainingArguments, CustomArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )
    logger.info("Check {}".format(training_args.output_dir))
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    model_args.model_name_or_path = glob.glob(model_args.model_name_or_path)[0]
    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("/mnt/liupeiyu/nlp_data/GLUE/glue.py", data_args.task_name)
    elif data_args.train_file.endswith(".csv"):
        # Loading a dataset from local csv files
        datasets = load_dataset(
            "csv",
            data_files={"train": data_args.train_file, "validation": data_args.validation_file},
        )
    else:
        # Loading a dataset from local json files
        datasets = load_dataset(
            "json",
            data_files={"train": data_args.train_file, "validation": data_args.validation_file},
        )
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    # pretrain_run_args = json.load(open(f"{model_args.model_name_or_path}/args.json", "r"))

    # def get_correct_ds_args(pretrain_run_args):
    #     ds_args = Namespace()

    #     for k, v in pretrain_run_args.items():
    #         setattr(ds_args, k, v)

    #     # to enable HF integration
    #     #         ds_args.huggingface = True
    #     return ds_args

    # ds_args = get_correct_ds_args(pretrain_run_args)

    # # in so, deepspeed is required
    # if (
    #     "deepspeed_transformer_kernel" in pretrain_run_args
    #     and pretrain_run_args["deepspeed_transformer_kernel"]
    # ):
    #     logger.warning("Using deepspeed_config due to kernel usage")

    #     remove_cuda_compatibility_for_kernel_compilation()
    model_cls, config_cls, token_cls = MODELS[custom_args.glue_model_type]
    if os.path.isdir(model_args.model_name_or_path):
        config = config_cls.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
        )
        custom_config = custom_args.__dict__
        if 'mpo_config' in custom_config:
            with open(custom_config['mpo_config'],'r') as fr:
                content = json.load(fr)
            for value_k, value_v in content.items():
                setattr(config, value_k, value_v)
        for k, v in custom_config.items():
            setattr(config, k, v)
        tokenizer = token_cls.from_pretrained(
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
        model = model_cls.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        config = config_cls.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        custom_config = custom_args.__dict__
        for k, v in custom_config.items():
            setattr(config, k, v)
        tokenizer = token_cls.from_pretrained(
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model = model_cls.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    logger.info(f"Before sharing: Total and trainable params: {str(get_parameter_number(model))}")
    # layer_idices = {0:[1,2,3,4,5,6,7,8,9,10,11]} # all share
    layer_idices = {0:list(range(1,config.num_hidden_layers))}
    # run_glue MPO
    if config.mpo_layers: # （1）not in load_layer：根据init的BERT权重，重新进行拆分。else：load已经训练好的MPObert权重。（2）删除非MPO层
        if ('FFN_1' in config.mpo_layers) and ('FFN_2' in config.mpo_layers):
            if ('FFN_1' not in config.load_layer) and ('FFN_2' not in config.load_layer):
                for i in range(config.num_hidden_groups):
                    model.albert.encoder.albert_layer_groups[i].albert_layers[0].from_pretrained_mpo()
            for i in range(config.num_hidden_groups):
                del model.albert.encoder.albert_layer_groups[i].albert_layers[0].ffn
                del model.albert.encoder.albert_layer_groups[i].albert_layers[0].ffn_output

        if 'attention' in config.mpo_layers:
            if 'attention' not in config.load_layer:
                for i in range(config.num_hidden_groups):
                    model.albert.encoder.albert_layer_groups[i].albert_layers[0].attention.from_pretrained_mpo()
            for i in range(config.num_hidden_groups):   
                del model.albert.encoder.albert_layer_groups[i].albert_layers[0].attention.query
                del model.albert.encoder.albert_layer_groups[i].albert_layers[0].attention.key
                del model.albert.encoder.albert_layer_groups[i].albert_layers[0].attention.value
                del model.albert.encoder.albert_layer_groups[i].albert_layers[0].attention.dense
    logger.info(f"\nAfter sharing: Total and trainable params: {str(get_parameter_number(model))}")
    layer_idices = {0:list(range(config.num_hidden_layers))}
    if layer_idices:
        names_tobe_shared = chain_module_names_mpoalbert("albert", layer_idices=layer_idices)
        model.set_names_tobe_shared(names_tobe_shared)
        model.replace(root_name="")
        del model.names_tobe_shared
    else:
        print(f"No parameter sharing")
    
    print(model)
    logger.info(f"\nAfter sharing: Total and trainable params: {str(get_parameter_number(model))}")
    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [
            name for name in datasets["train"].column_names if name != "label"
        ]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
        max_length = data_args.max_seq_length
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False
        max_length = None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [label_to_id[l] for l in examples["label"]]
        return result

    datasets = datasets.map(
        preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache
    )

    train_dataset = datasets["train"]
    eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
    if data_args.task_name is not None:
        test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("/home/liupeiyu/academic-budget-bert/pretraining/compress_tools_v2/glue.py", data_args.task_name)
    # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
    # compute_metrics

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    try:
        import wandb

        wandb.init(
            project=training_args.project_name,
            group=training_args.group_name,
            name=config.runname,
            dir="/tmp",
        )
        wandb.config.update(model_args)
        wandb.config.update(data_args)
        wandb.config.update(training_args)

    except Exception as e:
        logger.warning("W&B logger is not available, please install to get proper logging")
        logger.error(e)

    # update state.json
    try:
        pass
        # os.popen(f"echo {wandb.run.id} {custom_args.runname} {data_args.task_name} {model_args.model_name_or_path} {custom_args.log_file_path} >> '/home/liupeiyu/academic-budget-bert/run_seq_exp/pipe.txt'")
    except e:
        logger.info(f"Check Wandb Error {e}")

    # init early stopping callback and metric to monitor
    callbacks = None
    if training_args.early_stopping_patience > 0:
        early_cb = EarlyStoppingCallback(training_args.early_stopping_patience)
        callbacks = [early_cb]

        metric_monitor = {
            "mrpc": "f1",
            "sst2": "accuracy",
            "mnli": "accuracy",
            "mnli_mismatched": "accuracy",
            "mnli_matched": "accuracy",
            "cola": "matthews_correlation",
            "stsb": "spearmanr",
            "qqp": "f1",
            "qnli": "accuracy",
            "rte": "accuracy",
            "wnli": "accuracy",
        }
        metric_to_monitor = metric_monitor[data_args.task_name]
        setattr(training_args, "metric_for_best_model", metric_to_monitor)

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=callbacks,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        trainer.train()

    # Evaluation
    if training_args.do_eval:
        print("*** Evaluate ***")
        metrics = trainer.evaluate()
        # trainer.log_metrics("eval", metrics)
        try:
            wandb.run.summary.update(metrics)
            log_metrics = {}
            for k, v in metrics.items():
                log_metrics["final_" + k] = v
            wandb.log(log_metrics)
        except Exception as e:
            logger.warning("W&B logger is not available, please install to get proper logging")
            logger.error(e)

    if training_args.do_predict:
        logger.info("*** Test ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(datasets["test_mismatched"])

        for test_dataset, task in zip(test_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            # test_dataset.remove_columns_("label")
            test_dataset = test_dataset.remove_columns("label")
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            predictions = (
                np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)
            )

            test_results_file_name = f"test_results_{task}_{unique_run_id}.txt"
            if os.path.isdir(model_args.model_name_or_path):
                output_test_file = os.path.join(
                    model_args.model_name_or_path, test_results_file_name
                )
            else:
                output_test_file = os.path.join(training_args.output_dir, test_results_file_name)

            print(f"test_results_file_name: {test_results_file_name}")

            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info(f"***** Test results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            if task in ['qnli','mnli','rte','mnli-mm']:
                                item = label_list[item]
                            writer.write(f"{index}\t{item}\n")


if __name__ == "__main__":
    main()
