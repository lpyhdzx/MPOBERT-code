from functools import partial
import sys
sys.path.append("/home/liupeiyu/MPO-albert_cuda/pretraining")
import numpy as np
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
# import torchvision
# import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import ray
from ray.tune.examples.pbt_transformers.utils import build_compute_metrics_fn
from ray.tune.schedulers import PopulationBasedTraining

######################################################################
from task_config import *
from task_config import task_name, tune_task, task_metric
from transformers import glue_tasks_num_labels, GlueDataset, \
    GlueDataTrainingArguments, TrainingArguments,PretrainedConfig
from transformers import AutoTokenizer
from pretraining.compress_tools_v2.trainer_bert_4 import Trainer
from pretraining.albert import AlbertForSequenceClassification
from pretraining.configs import PretrainedAlbertConfig
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.utils.data.dataloader import DataLoader
from transformers.trainer_utils import default_compute_objective
from pretraining.sharer import chain_module_names_mpoalbert
from datasets import load_dataset

import logging
logger = logging.getLogger(__name__)

def get_parameter_number(net):
    '''
    :param net: model class
    :return: params statistics
    '''
    total_num = sum(p.numel() for p in net.parameters())/1000/1000
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)/1000/1000
    return {'Total(M)': total_num, 'Trainable(M)': trainable_num}

def load_data(task_name, data_dir, tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if task_name == 'sst-2':
        datasets = load_dataset("/mnt/liupeiyu/nlp_data/GLUE/glue.py", 'sst2')
        sentence1_key, sentence2_key = task_to_keys['sst2']
    else:
        datasets = load_dataset("/mnt/liupeiyu/nlp_data/GLUE/glue.py", task_name)
        sentence1_key, sentence2_key = task_to_keys[task_name]
    
    if task_name == 'stsb':
        num_labels = glue_tasks_num_labels['sts-b']
    elif task_name == 'sst2':
        num_labels = glue_tasks_num_labels['sst-2']
    else:
        num_labels = glue_tasks_num_labels[task_name]
    model_config = PretrainedAlbertConfig.from_pretrained(
        model_name, num_labels=num_labels, finetuning_task=task_name)
    label_to_id = None
    label2id = model_config.label2id
    is_regression = task_name == "stsb"
    if not is_regression:
        label_list = datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1
    if (
        label2id != PretrainedConfig(num_labels=num_labels).label2id
        and task_name is not None
        and is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding="max_length", max_length=128, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [label_to_id[l] for l in examples["label"]]
        return result

    datasets = datasets.map(
        preprocess_function, batched=True, load_from_cache_file=True
    )

    train_dataset = datasets["train"]
    eval_dataset = datasets["validation_matched" if task_name == "mnli" else "validation"]
    return train_dataset, eval_dataset

def create_optimizer(model, learning_rate, 
                    adam_beta1=0.9, adam_beta2=0.9, adam_epsilon=0.99,weight_decay=0.0467984,warmup_steps=20,mpo_lr=-1):
    """
    Setup the optimizer and the learning rate scheduler.

    We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
    """
    no_decay = ["bias", "LayerNorm.weight"]
    mpo_layer = [p for n, p in model.named_parameters() if "tensor_set" in n]
    
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and not "tensor_set" in n],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and not "tensor_set" in n],
            "weight_decay": 0.0,
        },
        {
            "params": mpo_layer,
            "lr": mpo_lr if mpo_lr > 0 else learning_rate,
            "weight_decay": weight_decay
        }
    ]
    assert model.num_parameters() == sum([sum([j.numel() for j in i['params']]) for i in optimizer_grouped_parameters])

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        eps=adam_epsilon,
    )
    return optimizer


def model_init():
    if task_name == 'stsb':
        num_labels = glue_tasks_num_labels['sts-b']
    elif task_name == 'sst2':
        num_labels = glue_tasks_num_labels['sst-2']
    else:
        num_labels = glue_tasks_num_labels[task_name]
    config = PretrainedAlbertConfig.from_pretrained(
        model_name, num_labels=num_labels, finetuning_task=task_name)
    config.mpo_layers = mpo_layers
    config.load_layer = load_layer
    config.emb_trunc = emb_trunc
    config.linear_trunc = linear_trunc
    config.attention_trunc = attention_trunc
    config.embed_size = 30000
    # config.mpo_lr = 2e-5
    config.max_seq_length = 128
    config.lora_linear = args.lora_linear
    with open("/home/liupeiyu/academic-budget-bert/mpo_shape_config.json",'r') as fr:
        content = json.load(fr)
    for value_k, value_v in content.items():
        setattr(config, value_k, value_v)
    model = AlbertForSequenceClassification.from_pretrained(
        model_name,
        from_tf=bool(".ckpt" in model_name),
        config=config,
        cache_dir=None
    )
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
    print("Total Parameter Count: {}M".format(model.num_parameters()/1000/1000))
    print("Total and trainable params: {}".format(str(get_parameter_number(model))))
        # return model
    layer_idices = {0:list(range(config.num_hidden_layers))}
    # layer_idices = {}
    if layer_idices:
        names_tobe_shared = chain_module_names_mpoalbert("albert", layer_idices=layer_idices)
        model.set_names_tobe_shared(names_tobe_shared)
        model.replace(root_name="")
        del model.names_tobe_shared
    else:
        print(f"No parameter sharing")
    print(f"\nAfter sharing: Total and trainable params: {str(get_parameter_number(model))}")
    return model

def train_finetune(config, checkpoint_dir=None, data_dir=None):
    net = model_init()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    train_dataset, eval_dataset = load_data(task_name, data_dir, model_name)

# training argumanets
    os.makedirs(checkpoint_path, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=checkpoint_path,
        learning_rate=config["learning_rate"],  # config
        do_train=True,
        do_eval=True,
        evaluation_strategy='steps',
        eval_steps=(config["num_train_epochs"]*len(train_dataset) // config["per_device_train_batch_size"] // config["gradient_accumulation_steps"])//6 + 1,
        save_steps=(config["num_train_epochs"]*len(train_dataset) // config["per_device_train_batch_size"] // config["gradient_accumulation_steps"])//6 + 1,
        num_train_epochs=config["num_train_epochs"],  # config
        max_steps=-1,
        per_device_train_batch_size=config["per_device_train_batch_size"],  # config
        per_device_eval_batch_size=8,  # config
        warmup_steps=config["warmup_steps"],
        weight_decay=config["weight_decay"],  # config
        logging_dir="./log_pbt",
        gradient_accumulation_steps = config["gradient_accumulation_steps"],
        load_best_model_at_end=True
    )
    training_args.mpo_lr = config["mpo_lr"]

    trainer = Trainer(
        model=net,
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(task_name) if task_name != 'stsb' else build_compute_metrics_fn("sts-b"))
        # optimizers=(optimizer, None))  # tokenizer=tokenizer, # transformer==4.7.0
    trainer.train()
    metrics = trainer.evaluate()
    objective = default_compute_objective(metrics)
    # if checkpoint_dir:
    # trainer._tune_save_checkpoint()
    tune.report(objective=objective, **metrics, done=True)

def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2, best_config=None):
    if not best_config:
        tune_config = eval(tune_task+'_config') if not tune_task == 'sst-2' else eval('sst2'+'_config')
    else:
        tune_config = best_config
        num_samples = 1

    scheduler = PopulationBasedTraining(
                    time_attr="training_iteration",metric=task_metric[tune_task],mode="max",perturbation_interval=1,hyperparam_mutations=tune_config)
    reporter = CLIReporter(
        parameter_columns={
            "weight_decay": "w_decay",
            "learning_rate": "lr",
            "per_device_train_batch_size": "train_bs/gpu",
            "num_train_epochs": "num_train_epochs",
            "warmup_steps":"warmup_steps",
            "gradient_accumulation_steps":"gradient_accumulation_steps",
            "mpo_lr":"mpo_lr",
            "seed":"seed"
        },
        metric_columns=[
            task_metric[tune_task], "eval_loss", "epoch", "training_iteration","eval_f1","eval_acc","eval_spearmanr","eval_mnli/acc"
        ])

    result = tune.run(
        tune.with_parameters(train_finetune, data_dir=data_dir),
        resources_per_trial={"gpu": gpus_per_trial},
        config=tune_config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        checkpoint_score_attr="eval_acc",
        keep_checkpoints_num=1)

    best_trial = result.get_best_trial("eval_acc", "max", "all")
    print(best_trial)


if __name__ == "__main__":
    if args.best_trial:
        best_config = eval(f"{task_name}_best_config") if task_name != 'sst-2' else eval(f"sst2_best_config")
        main(num_samples=1, max_num_epochs=3, gpus_per_trial=1, best_config=best_config)
    else:
        main(num_samples=48, max_num_epochs=3, gpus_per_trial=1)