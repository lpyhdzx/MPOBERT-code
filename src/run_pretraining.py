# coding=utf-8
# Copyright 2021 Intel Corporation. All rights reserved.
# code taken from commit: 35b4582486fe096a5c669b6ca35a3d5c6a1ec56b
# https://github.com/microsoft/DeepSpeedExamples/tree/master/bing_bert
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

import json
import logging
import os
import random
import time
from argparse import Namespace
from pretraining.args.dataset_args import PreTrainDatasetArguments
from pretraining.args.deepspeed_args import DeepspeedArguments
from pretraining.args.model_args import ModelArguments, ModelConfigArguments
from pretraining.args.optimizer_args import OptimizerArguments
from pretraining.args.pretraining_args import PretrainScriptParamsArguments
from pretraining.args.scheduler_args import SchedulerArgs
from pretraining.base import BasePretrainModel
from pretraining.dataset.distributed_pretraining_dataset import (
    PreTrainingDataset as DistPreTrainingDataset,
)
from pretraining.dataset.pretraining_dataset import PreTrainingDataset, ValidationDataset
from pretraining.optimizers import get_optimizer
from pretraining.schedules import get_scheduler
from pretraining.utils import (
    Logger,
    get_time_diff_hours,
    is_time_to_exit,
    is_time_to_finetune,
    master_process,
    set_seeds,
)
from timeit import default_timer as get_now

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm
from transformers import HfArgumentParser
from pretraining.compress_tools_v2.custom_mpo_args import CustomArguments
# from nvitop.core import Device

# device = Device(0)
logger = Logger(cuda=torch.cuda.is_available())
# from gpu_mem_track import MemTracker
# gpu_tracker = MemTracker()
_has_wandb = False
try:
    import wandb

    _has_wandb = True
except:
    logger.warning(
        "W&B logger is not installed, \
        for advanced logging please install using pip install wandb"
    )

# torch.backends.cuda.matmul.allow_tf32 = True
# global_forward_time = 0.
# global_backward_time = 0.

global_step = 0
global_data_samples = 0


def get_valid_dataloader(args, dataset: Dataset):
    if args.local_rank == -1:
        train_sampler = RandomSampler(dataset)
    else:
        train_sampler = DistributedSampler(dataset)
    return (
        x
        for x in DataLoader(
            dataset,
            batch_size=args.validation_micro_batch,
            sampler=train_sampler,
            num_workers=0,
            pin_memory=True,
        )
    )


validation_shard_index = 0


def pretrain_validation(args, model, validation_dataset, step):
    global validation_shard_index

    logger.info(f"Validation micro batch size: {args.validation_micro_batch}")
    index = validation_shard_index
    validation_shard_index += 1
    model.eval()
    dataset = validation_dataset.get_validation_set(index)
    data_batches = get_valid_dataloader(args, dataset)
    eval_loss = 0
    num_eval_steps = 0
    with torch.no_grad():
        for _, batch in enumerate(tqdm(data_batches, smoothing=1)):
            batch = tuple(t.to(args.device) for t in batch)
            total_loss = model.forward(batch)

            torch.cuda.synchronize()
            # using all_reduce is IMPORTANT! it ensures validation loss consistency across all threads
            dist.all_reduce(total_loss)
            total_loss = total_loss / dist.get_world_size()
            eval_loss += total_loss.mean().item()
            num_eval_steps += 1
    eval_loss = eval_loss / num_eval_steps

    logger.info(f"Validation Loss for epoch/step {index + 1}/{step} is: {eval_loss}")
    if master_process(args):
        if _has_wandb:
            log_info = {
                f"Validation/Loss": eval_loss,
            }
            wandb.log(log_info, step=step)
    del dataset
    del data_batches
    del batch
    model.train()
    return eval_loss


def create_finetune_job(args, index, global_step, model):
    try:

        checkpoint_id = f"epoch{index}_step{global_step}"
        model.save_weights(
            checkpoint_id=checkpoint_id,
            output_dir=args.saved_model_path,
            is_deepspeed=args.deepspeed,
        )
        logger.info("Saved fine-tuning job.")
    except Exception as e:
        logger.warning("Finetune checkpoint failed.")
        logger.warning(e)


def train(
    args, index, model, optimizer, lr_scheduler, pretrain_dataset_provider, validation_dataset=None
):
    global global_step
    global global_data_samples

    dataset_iterator, total_length = pretrain_dataset_provider.get_shard(index)
    current_data_sample_count = global_data_samples

    logger.info(
        f"worker-{dist.get_rank()}: begin epoch {index} current_sample_count {current_data_sample_count} shard_length {total_length} global_data_samples {global_data_samples}"
    )

    pretrain_dataset_provider.prefetch_shard(index + 1)

    model.train()

    all_step_time = 0.0
    eval_loss = None
    scale_counter_at_1 = 0
    # logger.info(f"train function: before dataloader Mem: {device.memory_used_human()}， Util: {device.gpu_utilization_string()}")

    # global global_forward_time, global_backward_time

    for batch_index_number, batch_index in enumerate(tqdm(dataset_iterator, smoothing=1)):

        if batch_index_number > args.max_steps_per_epoch:
            logger.info("Max steps per epochs reached. Resuming to next epoch ...")
            break

        if batch_index_number >= len(dataset_iterator) - 1:
            # skip last batch
            continue

        try:
            step_start = time.time()

            batch = pretrain_dataset_provider.get_batch(batch_index)
            # logger.info(f"before to device Mem: {device.memory_used_human()}， Util: {device.gpu_utilization_string()}")
            batch = tuple(t.to(args.device) for t in batch)  # Move to GPU 这里用“-1” 表示不做CEloss的值，代替了-100
            # logger.info(f"after to device: before dataloader Mem: {device.memory_used_human()}， Util: {device.gpu_utilization_string()}")

            # torch.cuda.synchronize()
            # forward_start = time.time()
            total_loss = model.forward(batch)
            # torch.cuda.synchronize()
            # global_forward_time += time.time() - forward_start

            unscaled_loss = total_loss.item()
            current_data_sample_count += args.train_micro_batch_size_per_gpu * dist.get_world_size()

            # Prefetch training data
            pretrain_dataset_provider.prefetch_batch()

            # torch.cuda.synchronize()
            # backward_start = time.time()
            model.network.backward(total_loss)
            # torch.cuda.synchronize()
            # global_backward_time += time.time() - backward_start

            total_loss = None

            if model.network.is_gradient_accumulation_boundary():
                report_metrics(
                    args,
                    lr_scheduler.get_last_lr(),
                    unscaled_loss,
                    global_step,
                    current_data_sample_count,
                )

                model.network.step()
                global_step += 1

                # HACK: add to scale counter if stuck at scale 1 (to detect possible NaN (diverged model))
                if args.fp16 and optimizer.cur_scale == 1:
                    scale_counter_at_1 += 1
                    logger.info(f"Optimizer scale=={scale_counter_at_1}")

                if scale_counter_at_1 >= args.scale_cnt_limit:
                    logger.warning("Optimizer scale==1 counter has been reached")
                    del batch
                    break
            else:
                # Call DeepSpeed engine step on micro steps
                model.network.step()

        except StopIteration:
            continue

        step_time = time.time() - step_start
        all_step_time += step_time
        if (
            global_step % args.log_throughput_every == 0
            and global_step != 0
            and model.network.is_gradient_accumulation_boundary()
            and dist.get_rank() == 0
        ):
            one_step_bs = (
                args.train_micro_batch_size_per_gpu
                * args.gradient_accumulation_steps
                * dist.get_world_size()
                * args.log_throughput_every
            )
            logger.info(
                "At step {}, the throughput is {:2f} Samples/s".format(
                    global_step * args.gradient_accumulation_steps, one_step_bs / all_step_time
                )
            )
            all_step_time = 0.0

        del batch

        # if batch_index_number == 127:
        #     # break
        #     raise ValueError
    # return None, None

    torch.cuda.synchronize()
    dist.barrier(model.network.data_parallel_group)

    pretrain_dataset_provider.release_shard(index)
    global_data_samples = current_data_sample_count

    logger.info(f"Epoch {index}: check whether to run validation...")
    if validation_dataset is not None and scale_counter_at_1 < args.scale_cnt_limit:
        time_diff = get_time_diff_hours(get_now(), args.exp_start_marker)
        if should_run_validation(time_diff, args, epoch=index):
            eval_loss = pretrain_validation(args, model, validation_dataset, global_step)

    logger.info(f"Epoch {index}: check if time to save a fine-tune checkpoint")
    if (
        is_time_to_finetune(
            get_now(),
            args.exp_start_marker,
            args.finetune_time_markers,
            args.total_training_time,
        )
        and master_process(args)
        and scale_counter_at_1 < args.scale_cnt_limit
    ):
        logger.info("Creating a Fine-tune job")
        create_finetune_job(args, index, global_step, model)
    return eval_loss, scale_counter_at_1


def should_run_validation(time_diff, args, epoch):
    time_proportion = time_diff / args.total_training_time

    should_do_validation = False

    # is in first stage of training
    if time_proportion < args.validation_begin_proportion:
        should_do_validation = epoch % args.validation_epochs_begin == 0

    # is in last stage of training
    elif time_proportion > 1 - args.validation_end_proportion:
        should_do_validation = epoch % args.validation_epochs_end == 0

    # is in the middle stage of training
    else:
        should_do_validation = epoch % args.validation_epochs == 0

    return should_do_validation


def report_metrics(args, lr, loss, step, data_sample_count):
    current_lr = lr[0] if type(lr) == list else lr
    if master_process(args):
        if _has_wandb:
            log_info = {
                f"train/lr": current_lr,
                f"train/train_loss": loss,
            }
            wandb.log(log_info, step=step)
            samp_info = {
                f"Train/Samples/train_loss": loss,
                f"Train/Samples/lr": current_lr,
                f"Train/total_samples": data_sample_count,
            }
            wandb.log(samp_info, commit=False)

    if (step + 1) % args.print_steps == 0 and master_process(args):
        logger.info(
            f"pre-training progress: step={step + 1}, loss={loss}, lr={current_lr}, sample_count={data_sample_count}"
        )


def merge_args(arg_list):
    args = Namespace()
    for cur_args in arg_list:
        for key, value in cur_args.__dict__.items():
            setattr(args, key, value)
    return args


def get_arguments():
    parser = HfArgumentParser(
        (
            DeepspeedArguments,
            ModelArguments,
            ModelConfigArguments,
            PreTrainDatasetArguments,
            OptimizerArguments,
            PretrainScriptParamsArguments,
            SchedulerArgs,
            CustomArguments,
        )
    )

    (
        ds_args,
        model_args,
        model_config_args,
        dataset_args,
        optimizer_args,
        train_args,
        schedule_args,
        custom_args,
    ) = parser.parse_args_into_dataclasses()

    custom_config = custom_args.__dict__
    if 'mpo_config' in custom_config:
        with open(custom_config['mpo_config'],'r') as fr:
            content = json.load(fr)
        for value_k, value_v in content.items():
            setattr(model_config_args, value_k, value_v)
    for k, v in custom_config.items():
        setattr(model_config_args, k, v)
    args = merge_args([ds_args, model_args, dataset_args, train_args])
    args.model_config = vars(model_config_args)
    args.optimizer_args = optimizer_args
    args.schedule_args = schedule_args
    return args


def create_ds_config(args):
    """Create a Deepspeed config dictionary"""
    ds_config = {
        "train_batch_size": args.train_batch_size,
        "train_micro_batch_size_per_gpu": args.train_micro_batch_size_per_gpu,
        "steps_per_print": args.steps_per_print,
        "gradient_clipping": args.gradient_clipping,
        "wall_clock_breakdown": args.wall_clock_breakdown,
    }
    if args.model_config["zero"]:
        model_hidden_size = args.model_config["hidden_size"]
        # optimizer_config = {
        #     "stage": 2,
        #     "offload_param": {
        #         "device": "cpu",
        #         "pin_memory": True
        #     },
        #     "offload_optimizer": {
        #         "device": "cpu",
        #         "pin_memory": True
        #     },
        #     "allgather_partitions": True,
        #     "allgather_bucket_size": 1e8,
        #     "overlap_comm": True,
        #     "reduce_scatter": True,
        #     "reduce_bucket_size": 1e8,
        #     "contiguous_gradients": True
        # }
        optimizer_config = {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        }
        ds_config.update({"zero_optimization": optimizer_config})
        # ds_config.update({"zero_allow_untested_optimizer": True})

    if False:
        flops_profiler = {
            "enabled": True,
            "profile_step": 1,
            "module_depth": -1,
            "top_modules": 1,
            "detailed": False, # True,
            "output_file": None,
        }
        ds_config.update({"flops_profiler": flops_profiler})

    if args.prescale_gradients:
        ds_config.update({"prescale_gradients": args.prescale_gradients})

    if args.gradient_predivide_factor is not None:
        ds_config.update({"gradient_predivide_factor": args.gradient_predivide_factor})

    if args.fp16:
        if "ds" in args.fp16_backend:
            fp16_dict = {
                "enabled": True,
                "loss_scale": 0,
                "min_loss_scale": 1,
                "loss_scale_window": 1000,
                "hysteresis": 2,
            }
            ds_config.update({"fp16": fp16_dict})
        elif "apex" in args.fp16_backend:
            amp_dict = {
                "enabled": True,
                "opt_level": args.fp16_opt,
                "keep_batchnorm_fp32": True,
            }
            ds_config.update({"amp": amp_dict})

    return ds_config


def parse_arguments():
    """Parse all the arguments needed for the training process"""
    args = get_arguments()
    set_seeds(args.seed)
    args.logger = logger
    args.ds_config = create_ds_config(args)
    args.deepspeed_config = args.ds_config
    args.job_name = f"{args.job_name}-{args.current_run_id}"
    logger.info(f"Running Config File: {args.job_name}")
    logger.info(f"Args = {args}")
    os.makedirs(args.output_dir, exist_ok=True)
    args.saved_model_path = os.path.join(args.output_dir, args.job_name, args.current_run_id)
    return args


def prepare_optimizer_parameters(args, model):
    param_optimizer = list(model.network.named_parameters())
    param_optimizer = [n for n in param_optimizer if "pooler" not in n[0]]
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": args.optimizer_args.weight_decay,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def get_parameter_number(net):
    '''
    :param net: model class
    :return: params statistics
    '''
    total_num = sum(p.numel() for p in net.parameters())/1000/1000
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)/1000/1000
    return {'Total(M)': total_num, 'Trainable(M)': trainable_num}

def prepare_model_and_optimizer(args):
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)
    # Load Pre-training Model skeleton + supplied model config
    model = BasePretrainModel(args)

    print(f"Total Parameter Count: {model.network.num_parameters()/1000/1000}M")
    print(f"Total and trainable params: {str(get_parameter_number(model.network))}")

    # Optimizer parameters
    args.exp_start_marker = get_now()
    logger.info(f"Check exp_start_marker: {get_now()}")
    optimizer_grouped_parameters = model.prepare_optimizer_parameters(
        args.optimizer_args.weight_decay, args.model_config["CT_lr"]
    )
    optimizer = get_optimizer(args.optimizer_args, args.lr, optimizer_grouped_parameters)
    lr_scheduler = get_scheduler(args.schedule_args, optimizer, args)

    # DeepSpeed initializer handles FP16, distributed, optimizer automatically.
    model.network, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model.network,
        model_parameters=optimizer_grouped_parameters,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config_params=args.ds_config,
    )
    logger.info(f"optimizer type: {type(optimizer)}")
    logger.info(f"optimizer description: {optimizer}")

    # Overwrite application configs with DeepSpeed config
    args.train_micro_batch_size_per_gpu = model.network.train_micro_batch_size_per_gpu()
    args.gradient_accumulation_steps = model.network.gradient_accumulation_steps()

    # Set DeepSpeed info
    args.local_rank = model.network.local_rank
    args.device = model.network.device
    args.fp16 = model.network.fp16_enabled()

    return model, optimizer, lr_scheduler


def check_if_early_stop(eval_loss, scale_counter, args):
    # check if the validation loss is already NaN and stop
    if eval_loss is not None and np.isnan(eval_loss):
        return True

    if scale_counter >= args.scale_cnt_limit:
        return True

    time_diff = get_time_diff_hours(get_now(), args.exp_start_marker)
    time_diff_minutes = time_diff * 60

    loss_to_compare_to = args.early_stop_eval_loss
    eval_loss_too_high = (eval_loss is not None) and (eval_loss > loss_to_compare_to)

    # if enough time passed, and the validation loss is not low enough, stop the run
    should_stop = time_diff_minutes > args.early_stop_time and eval_loss_too_high

    logger.info(
        json.dumps(
            {
                "time_diff_minutes": time_diff_minutes,
                "loss_to_compare_to": loss_to_compare_to,
                "eval_loss": eval_loss,
                "should_stop": should_stop,
            }
        )
    )

    return should_stop


def load_datasets(args):
    if "per_device" in args.data_loader_type:
        train_ds = PreTrainingDataset(args, logger=args.logger)
    else:
        train_ds = DistPreTrainingDataset(args, logger=args.logger)
    valid_ds = ValidationDataset(args) if args.do_validation else None
    return train_ds, valid_ds


def start_training(args, model, optimizer, lr_scheduler, start_epoch):
    """Training loop (epochs, and detect points of exit)"""
    global global_step
    global global_data_samples

    pretrain_dataset_provider, validation_dataset = load_datasets(args)

    last_epoch = 0
    print(f'num_epochs = {args.num_epochs}')
    for index in range(start_epoch, args.num_epochs):
        last_epoch = index
        logger.info(f"Training Epoch: {index}")
        pre = time.time()

        eval_loss, scale_counter = train(
            args,
            index,
            model,
            optimizer,
            lr_scheduler,
            pretrain_dataset_provider,
            validation_dataset,
        )

        post = time.time()
        logger.info(f"Total time for epoch {index}: {post-pre} seconds")

        should_early_stop = (
            check_if_early_stop(eval_loss, scale_counter, args)
            if args.use_early_stopping
            else False
        )

        # check if training reached a stopping point
        if is_time_to_exit(get_now(), args=args, global_steps=global_step) or should_early_stop:
            logger.info(
                f"Warning: Early training termination due to max steps limit or time limit, \
                    epoch={index}, global_step={global_step}"
            )
            break

        # save a checkpoint
        if (
            index > 0
            and args.num_epochs_between_checkpoints > 0
            and index % args.num_epochs_between_checkpoints == 0
        ):
            logger.info(f"Process rank - {dist.get_rank()} - attempting to save checkpoint")
            save_training_checkpoint(
                model,
                model_path=args.saved_model_path,
                epoch=index + 1,
                last_global_step=global_step,
                last_global_data_samples=global_data_samples,
                exp_start_marker=args.exp_start_marker,
                # ckpt_id="latest_checkpoint", # None 则以step为标识保存多个checkpoint，latest_checkpoint 则会一直被最新的覆盖
            )
            dist.barrier()
    logger.info(
        "Training is complete or training limit has been reached.\
            Proceeding with checkpointing/validation"
    )

    # save a fine-tune checkpoint
    if master_process(args) and args.finetune_checkpoint_at_end:
        create_finetune_job(args, args.num_epochs, global_step, model)

    save_training_checkpoint(
        model,
        model_path=args.saved_model_path,
        epoch=last_epoch + 1,
        last_global_step=global_step,
        last_global_data_samples=global_data_samples,
        exp_start_marker=args.exp_start_marker,
        ckpt_id="latest_checkpoint",
    )

    logger.info("Waiting for all processes (barrier)")
    torch.cuda.synchronize()
    dist.barrier()
    logger.info("All nodes/processes are synced, proceed to exit")

    # run a final validation check
    _ = pretrain_validation(args, model, validation_dataset, global_step)
    logger.info("Final validation results computed")


def setup_wandb(args, model, resume_id=None):
    if _has_wandb and master_process(args):
        if resume_id is not None:
            wandb.init(
                project=args.project_name,
                group=args.job_name,
                dir="/tmp",
                resume="allow",
                id=resume_id,
            )
        else:
            wandb.init(project=args.project_name, group=args.job_name, dir="/tmp")
        wandb.config.update(args, allow_val_change=True)
        wandb.watch(model)
    else:
        logger.info("W&B library not installed. Using only CLI logging.")


def save_training_checkpoint(
    model,
    model_path,
    epoch,
    last_global_step,
    last_global_data_samples,
    exp_start_marker,
    ckpt_id=None,
    **kwargs,
):
    """Utility function for checkpointing model + optimizer dictionaries
    The main purpose for this is to be able to resume training from that instant again
    """
    checkpoint_state_dict = {
        "epoch": epoch,
        "last_global_step": last_global_step,
        "last_global_data_samples": last_global_data_samples,
        "exp_time_marker": get_now() - exp_start_marker,  ## save total training time in seconds
    }
    if _has_wandb and dist.get_rank() == 0:
        checkpoint_state_dict.update({"run_id": wandb.run.id})
    # Add extra kwargs too
    checkpoint_state_dict.update(kwargs)

    status_msg = "checkpointing training model: PATH={}, ckpt_id={}".format(model_path, ckpt_id)
    # save_checkpoint is DS method
    success = model.network.save_checkpoint(
        model_path, tag=ckpt_id, client_state=checkpoint_state_dict
    )
    if success:
        logging.info(f"Success {status_msg}")
    else:
        logging.warning(f"Failure {status_msg}")
    return


def load_training_checkpoint(model, model_path, ckpt_id, warmup=False, zero=False):
    """Utility function for checkpointing model + optimizer dictionaries
    The main purpose for this is to be able to resume training from that instant again
    """
    if warmup:
        logger.info("===== Check using {} weight only without optimizer and lr_scheduler for warmup =====".format(model_path))
        if zero:
            checkpoint_state_dict = torch.load("{}/epoch1000000_step2119/pytorch_model.bin".format(model_path), map_location=lambda storage, loc: storage)
            logger.info("Check using deepspeed")
        else:
            checkpoint_state_dict = torch.load("{}/latest_checkpoint/mp_rank_00_model_states.pt".format(model_path), map_location=lambda storage, loc: storage)
        moduel = checkpoint_state_dict['module'] if hasattr(checkpoint_state_dict, "module") else checkpoint_state_dict
        model.network.load_module_state_dict(moduel, strict=False)
        # without warmup record
        epoch = 0
        last_global_step = 0
        last_global_data_samples = 0
        total_seconds_training = 0
        wandb_run_id = None
        del checkpoint_state_dict
        return (epoch, last_global_step, last_global_data_samples, total_seconds_training, wandb_run_id)
    else:
        _, checkpoint_state_dict = model.network.load_checkpoint(
            model_path, ckpt_id,
            load_lr_scheduler_states=True # ALBERT set true; BERT set false 取决于每次run是否从上次最后的lr开始
        )  # load_checkpoint is DS method
        epoch = checkpoint_state_dict["epoch"]
        last_global_step = checkpoint_state_dict["last_global_step"]
        last_global_data_samples = checkpoint_state_dict["last_global_data_samples"]
        total_seconds_training = checkpoint_state_dict["exp_time_marker"]
        wandb_run_id = checkpoint_state_dict["run_id"]
    del checkpoint_state_dict
    return (epoch, last_global_step, last_global_data_samples, total_seconds_training, wandb_run_id)


def prepare_resuming_checkpoint(args, model):
    global global_step
    global global_data_samples

    logger.info(
        f"Restoring previous training checkpoint from PATH={args.load_training_checkpoint}, \
            CKPT_ID={args.load_checkpoint_id}"
    )
    (
        start_epoch,
        global_step,
        global_data_samples,
        training_time_diff,
        wandb_run_id,
    ) = load_training_checkpoint(
        model=model,
        model_path=args.load_training_checkpoint,
        ckpt_id=args.load_checkpoint_id,
        warmup=args.model_config["warmup"],
        zero=args.model_config["zero"]
    )
    logger.info(
        f"The model is loaded from last checkpoint at epoch {start_epoch} when the global steps \
            were at {global_step} and global data samples at {global_data_samples}"
    )
    # adjust the time trained according to training clock
    args.exp_start_marker = get_now() - training_time_diff

    return start_epoch, wandb_run_id


def main():
    start_time = time.time()
    args = parse_arguments()
    # args.exp_start_marker = get_now()
    model, optimizer, lr_scheduler = prepare_model_and_optimizer(args)

    start_epoch = 0
    wandb_run_id = None

    # Load a checkpoint if resuming training
    if args.load_training_checkpoint is not None:
        start_epoch, wandb_run_id = prepare_resuming_checkpoint(args, model)

    # setup W&B logging
    setup_wandb(args, model.network, resume_id=wandb_run_id)
    # logger.info(f"Mem: {device.memory_used_human()}， Util: {device.gpu_utilization_string()}")
    start_training(args, model, optimizer, lr_scheduler, start_epoch)

    end_time = time.time() - start_time
    logger.info(f"Training time: {end_time} seconds")
    logger.info("Training ends.")


if __name__ == "__main__":
    try:
        main()
    except ValueError:
        print(f'global_forward_time = {global_forward_time:.6f} s, global_backward_time = {global_backward_time:.6f} s')
    except Exception as e:
        raise e
