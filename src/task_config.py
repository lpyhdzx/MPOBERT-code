from ray import tune
import argparse
import os
from transformers import HfArgumentParser
from pretraining.compress_tools_v2.custom_mpo_args import CustomArguments

# parser = HfArgumentParser((CustomArguments))
# custom_args = parser.parse_args_into_dataclasses()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ray-address",
    type=str,
    default=None,
    help="Address to use for Ray. "
    "Use \"auto\" for cluster. "
    "Defaults to None for local.")
parser.add_argument("--task", type=str, default="")
parser.add_argument("--model_name", type=str, default="")
parser.add_argument("--runname", type=str, default="")
parser.add_argument("--lora_linear", action="store_true")
parser.add_argument("--best_trial", action="store_true",default=False)


args, _ = parser.parse_known_args()

runname = args.runname
checkpoint_path = os.path.join("/mnt/liupeiyu/checkpoint/mpobert/albert_pbt", runname)
# data argumenst
data_dir = '/mnt/liupeiyu/nlp_data/GLUE'
task_name = args.task
tune_task = 'stsb' if task_name == 'sts-b' else task_name
mpo_layers = 'FFN_1,FFN_2,attention'
load_layer = 'FFN_1,FFN_2,attention'

# model arguments
model_name = args.model_name
config_name = ""
tokenizer_name = ""
emb_trunc = 100000
linear_trunc = 100000
attention_trunc = 100000
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
# qqp_config = {
#     "weight_decay": tune.uniform(0.0, 0.3),
#     # "learning_rate": tune.uniform(2e-6, 2e-5), # 24 mpoalbert
#     # "mpo_lr": tune.uniform(2e-6, 2e-5),
#     "learning_rate": tune.uniform(2e-7, 2e-5), # 48 mpoalbert
#     "mpo_lr": tune.uniform(2e-7, 2e-5),
#     # "per_device_train_batch_size": tune.choice([16,32]), # 24 mpoalbert
#     "per_device_train_batch_size": tune.choice([16]), # 48 mpoalbert
#     "gradient_accumulation_steps": tune.choice([4,8]), # 48 mpoalbert
#     "warmup_steps":tune.choice([200,500]),
#     "num_train_epochs":tune.choice([3]),
#     "seed":tune.choice([42])
# }
qqp_config = {
    "weight_decay": tune.uniform(0.0, 0.1),
    "learning_rate": tune.uniform(2e-8, 2e-6), # 48 mpoalbert
    "mpo_lr": tune.uniform(2e-8, 2e-6),
    "per_device_train_batch_size": tune.choice([16]),
    "gradient_accumulation_steps": tune.choice([4,8]), # 48 mpoalbert
    "warmup_steps":tune.choice([200,500]),
    "num_train_epochs":tune.choice([3]),
    "seed":tune.choice([42])
}
qnli_config = {
    "weight_decay": tune.uniform(0.0, 0.3),
    # "learning_rate": tune.uniform(2e-6, 2e-5), # 24 mpoalbert
    # "mpo_lr": tune.uniform(2e-6, 2e-5),
    "learning_rate": tune.uniform(2e-7, 2e-5), # 48 mpoalbert
    "mpo_lr": tune.uniform(2e-7, 2e-5),
    # "per_device_train_batch_size": tune.choice([16,32]), # 24 mpoalbert
    "per_device_train_batch_size": tune.choice([16]), # 48 mpoalbert
    "gradient_accumulation_steps": tune.choice([4,8]), # 48 mpoalbert
    "warmup_steps":tune.choice([200,500]),
    "num_train_epochs":tune.choice([3]),
    "seed":tune.choice([42])
}
mnli_config = {
    "weight_decay": tune.uniform(0.0, 0.3),
    "learning_rate": tune.uniform(2e-6, 2e-4),
    "mpo_lr": tune.uniform(2e-6, 2e-4),
    "per_device_train_batch_size": tune.choice([32]),
    "warmup_steps":tune.choice([200,500]),
    "num_train_epochs":tune.choice([3]),
    "seed":tune.choice([42,24])
}
mnli_config = {
    "weight_decay": tune.uniform(0.0, 0.3),
    # "learning_rate": tune.uniform(2e-6, 2e-5), # 24 mpoalbert
    # "mpo_lr": tune.uniform(2e-6, 2e-5),
    "learning_rate": tune.uniform(2e-7, 2e-5), # 48 mpoalbert
    "mpo_lr": tune.uniform(2e-7, 2e-5),
    # "per_device_train_batch_size": tune.choice([16,32]), # 24 mpoalbert
    "per_device_train_batch_size": tune.choice([16]), # 48 mpoalbert
    "gradient_accumulation_steps": tune.choice([4,8]), # 48 mpoalbert
    "warmup_steps":tune.choice([0,200,500]),
    "num_train_epochs":tune.choice([3]),
    "seed":tune.choice([42])
}
sst2_config = {
    "weight_decay": tune.uniform(0.0, 0.3),
    # "learning_rate": tune.uniform(2e-6, 2e-5), # 24 mpoalbert
    # "mpo_lr": tune.uniform(2e-6, 2e-5),
    "learning_rate": tune.uniform(2e-8, 2e-6), # 48 mpoalbert
    "mpo_lr": tune.uniform(2e-8, 2e-6),
    # "per_device_train_batch_size": tune.choice([16,32]), # 24 mpoalbert
    "per_device_train_batch_size": tune.choice([16]), # 48 mpoalbert
    "gradient_accumulation_steps": tune.choice([4,8]), # 48 mpoalbert
    "warmup_steps":tune.choice([0,500]),
    "num_train_epochs":tune.choice([3]),
    "seed":tune.choice([42])
}
rte_config = {
    "weight_decay": tune.uniform(0.0, 0.3),
    # "learning_rate": tune.uniform(2e-6, 2e-5), # 24 mpoalbert
    # "mpo_lr": tune.uniform(2e-6, 2e-5),
    "learning_rate": tune.uniform(2e-8, 2e-6), # 48 mpoalbert
    "mpo_lr": tune.uniform(2e-8, 2e-6),
    # "per_device_train_batch_size": tune.choice([16,32]), # 24 mpoalbert
    "per_device_train_batch_size": tune.choice([16]), # 48 mpoalbert
    "gradient_accumulation_steps": tune.choice([4,8]), # 48 mpoalbert
    "warmup_steps":tune.choice([0,20]),
    "num_train_epochs":tune.choice([5]),
    "seed":tune.choice([42])
}
stsb_config = {
    "weight_decay": tune.uniform(0.0, 0.3),
    "learning_rate": tune.uniform(2e-6, 2e-4),
    "mpo_lr": tune.uniform(2e-6, 2e-4),
    "per_device_train_batch_size": tune.choice([16,32]),
    "warmup_steps":tune.choice([0,100,200]),
    "num_train_epochs":tune.choice([5]),
    "seed":tune.choice([42])
}
cola_config = {
    "weight_decay": tune.uniform(0.0,0.1),
    # "learning_rate": tune.uniform(2e-6, 2e-5), # 24 mpoalbert
    # "mpo_lr": tune.uniform(2e-6, 2e-5),
    "learning_rate": tune.uniform(2e-8, 2e-6), # 48 mpoalbert
    "mpo_lr": tune.uniform(2e-8, 2e-6),
    # "per_device_train_batch_size": tune.choice([16,32]), # 24 mpoalbert
    "per_device_train_batch_size": tune.choice([16]), # 48 mpoalbert
    "gradient_accumulation_steps": tune.choice([1,2]), # 48 mpoalbert
    "warmup_steps":tune.choice([0]),
    "num_train_epochs":tune.choice([3,5]),
    "seed":tune.choice([42])
}
# cola_config = {
#     "weight_decay": tune.uniform(0.0,0.1),
#     # "learning_rate": tune.uniform(2e-6, 2e-5), # 24 mpoalbert
#     # "mpo_lr": tune.uniform(2e-6, 2e-5),
#     "learning_rate": tune.uniform(2e-7, 5e-6), # 48 mpoalbert
#     "mpo_lr": tune.uniform(2e-7, 5e-6),
#     # "per_device_train_batch_size": tune.choice([16,32]), # 24 mpoalbert
#     "per_device_train_batch_size": tune.choice([16]), # 48 mpoalbert
#     "gradient_accumulation_steps": tune.choice([4,8]), # 48 mpoalbert
#     "warmup_steps":tune.choice([0]),
#     "num_train_epochs":tune.choice([5]),
#     "seed":tune.choice([42])
# }
task_metric = {
    "qqp":'eval_acc',
    "qnli":'eval_acc',
    "mnli":'eval_mnli/acc',
    "rte":"eval_acc",
    "stsb":"eval_spearmanr",
    "mrpc":"eval_f1",
    "cola":"eval_mcc",
    "sst-2":"eval_acc"
}
