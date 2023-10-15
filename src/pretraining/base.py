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
import torch

from transformers import BertTokenizer, RobertaTokenizer, AlbertTokenizer
from pretraining.albert import AlbertForPreTraining

from pretraining.configs import PretrainedBertConfig, PretrainedRobertaConfig, PretrainedAlbertConfig, AlbertConfig
from pretraining.modeling import BertForPreTraining, BertLMHeadModel
# from pretraining.modeling_albert import BertForPreTraining as AlbertForPreTraining
# from pretraining.MPOP_albert import AlbertForPreTraining
from pretraining.utils import to_sanitized_dict
from pretraining.sharer import route_name_to_module, chain_module_names, _bert_roberta_names, _albert_roberta_names, chain_module_names_orialbert, chain_module_names_mpoalbert
from pretraining.compress_tools_v2.Matrix2MPO_beta import MPO


logger = logging.getLogger(__name__)


MODELS = {
    "bert-mlm": (BertLMHeadModel, PretrainedBertConfig, BertTokenizer),
    "bert-mlm-roberta": (BertLMHeadModel, PretrainedRobertaConfig, RobertaTokenizer),
    "bert-mlm-nsp": (BertForPreTraining, PretrainedBertConfig, BertTokenizer),
    "albert-mlm-sop": (AlbertForPreTraining, AlbertConfig, AlbertTokenizer),
    # "albert-mlm-sop": (AlbertForPreTraining, PretrainedAlbertConfig, AlbertTokenizer),
}

def model_from_pretrained_mpo(model, args):
    if args.model_config["load_from_albert"]:
        logger.info("===== Check loading weights from {} to BERT =====".format(args.model_config["load_from_albert"]))
        albert_weights = args.model_config["load_from_albert"]
        checkpoint_state_dict = torch.load(albert_weights)
        module = checkpoint_state_dict['module'] if hasattr(checkpoint_state_dict, "module") else checkpoint_state_dict
    
        # embeddings
        for shared_name in ["embeddings.word_embeddings", "embeddings.position_embeddings","embeddings.token_type_embeddings","encoder.embedding_hidden_mapping_in"]:
            init_module = module.get("albert." + shared_name + ".weight")
            target_module = eval("model.network.bert." + shared_name + ".weight")
            target_module.data = init_module.data
            if hasattr(target_module, "bias"):
                target_module.bias = init_module.bias
        if args.model_config["layernorm_embedding"]:
            model.network.bert.embeddings.LayerNorm.weight.data = module.albert.embeddings.LayerNorm.weight.data
            model.network.bert.embeddings.LayerNorm.bias.data = module.albert.embeddings.LayerNorm.bias.data
        # abbres=["K","Q","V","AO","I","O"]
        for i in range(args.model_config["num_hidden_layers"]):
            model.network.bert.encoder.layer[i].intermediate.dense.weight.data = module.get(_albert_roberta_names["I"]+".weight").data
            model.network.bert.encoder.layer[i].output.dense.weight.data = module.get(_albert_roberta_names["O"]+".weight").data
            model.network.bert.encoder.layer[i].attention.self.query.weight.data = module.get(_albert_roberta_names["Q"]+".weight").data
            model.network.bert.encoder.layer[i].attention.self.key.weight.data = module.get(_albert_roberta_names["K"]+".weight").data
            model.network.bert.encoder.layer[i].attention.self.value.weight.data = module.get(_albert_roberta_names["V"]+".weight").data
            model.network.bert.encoder.layer[i].attention.output.dense.weight.data = module.get(_albert_roberta_names["AO"]+".weight").data
        # pooler
        model.network.bert.pooler.dense_act.weight.data = module.get("albert.pooler.weight").data
        model.network.bert.pooler.dense_act.bias.data = module.get("albert.pooler.bias").data
        model.network.cls.predictions.bias.data = module.get("predictions.bias").data
        model.network.cls.predictions.LayerNorm.weight.data = module.get("predictions.LayerNorm.weight").data
        model.network.cls.predictions.LayerNorm.bias.data = module.get("predictions.LayerNorm.bias").data
        model.network.cls.predictions.dense.weight.data = module.get("predictions.dense.weight").data
        model.network.cls.predictions.dense.bias.data = module.get("predictions.dense.bias").data
        model.network.cls.predictions.decoder.weight.data = module.get("predictions.decoder.weight").data
        model.network.cls.predictions.decoder.bias.data = module.get("predictions.decoder.bias").data

        del module
    if args.model_config["mpo_layers"]:
        if 'FFN_1' in args.model_config["mpo_layers"]:
            if 'FFN_1' not in args.model_config["load_layer"]:
                for i in range(args.model_config["num_hidden_layers"]):
                    model.network.bert.encoder.layer[i].intermediate.from_pretrained_mpo()
            for i in range(args.model_config["num_hidden_layers"]):
                del model.network.bert.encoder.layer[i].intermediate.dense
            
        if 'FFN_2' in args.model_config["mpo_layers"]:
            if 'FFN_2' not in args.model_config["load_layer"]:
                for i in range(args.model_config["num_hidden_layers"]):
                    model.network.bert.encoder.layer[i].output.from_pretrained_mpo()
            for i in range(args.model_config["num_hidden_layers"]):
                del model.network.bert.encoder.layer[i].output.dense

        if 'attention' in args.model_config["mpo_layers"]:
            if 'attention' not in args.model_config["load_layer"]:
                for i in range(args.model_config["num_hidden_layers"]):
                    model.network.bert.encoder.layer[i].attention.self.from_pretrained_mpo()
                    model.network.bert.encoder.layer[i].attention.output.from_pretrained_mpo()
            for i in range(args.model_config["num_hidden_layers"]):   
                del model.network.bert.encoder.layer[i].attention.self.query
                del model.network.bert.encoder.layer[i].attention.self.key
                del model.network.bert.encoder.layer[i].attention.self.value
                del model.network.bert.encoder.layer[i].attention.output.dense
    return model
def albert_from_pretrained_mpo(model, args):
    # assert args.model_config["num_hidden_groups"] == args.model_config["num_hidden_layers"]
    if args.model_config["mpo_layers"]:
        if ('FFN_1' in args.model_config["mpo_layers"]) and ('FFN_2' in args.model_config["mpo_layers"]):
            if ('FFN_1' not in args.model_config["load_layer"]) and ('FFN_2' not in args.model_config["load_layer"]):
                for i in range(args.model_config["num_hidden_groups"]):
                    model.network.albert.encoder.albert_layer_groups[i].albert_layers[0].from_pretrained_mpo()
            for i in range(args.model_config["num_hidden_groups"]):
                del model.network.albert.encoder.albert_layer_groups[i].albert_layers[0].ffn
                del model.network.albert.encoder.albert_layer_groups[i].albert_layers[0].ffn_output

        if 'attention' in args.model_config["mpo_layers"]:
            if 'attention' not in args.model_config["load_layer"]:
                for i in range(args.model_config["num_hidden_groups"]):
                    model.network.albert.encoder.albert_layer_groups[i].albert_layers[0].attention.from_pretrained_mpo()
            for i in range(args.model_config["num_hidden_groups"]):   
                del model.network.albert.encoder.albert_layer_groups[i].albert_layers[0].attention.query
                del model.network.albert.encoder.albert_layer_groups[i].albert_layers[0].attention.key
                del model.network.albert.encoder.albert_layer_groups[i].albert_layers[0].attention.value
                del model.network.albert.encoder.albert_layer_groups[i].albert_layers[0].attention.dense
    return model
class BasePretrainModel(object):
    def __init__(
        self,
        args,
        model_type=None,
        model_name_or_path=None,
        tokenizer=None,
        config=None,
        model_kwargs={},
    ):
        if not model_type:
            # getting default model type from args
            model_type = args.model_type
        assert model_type in MODELS, f"model_type {model_type} is not supported"
        model_cls, config_cls, token_cls = MODELS[model_type]

        self.args = args
        self.ds_file = args.ds_config if hasattr(args, "ds_config") else None

        if not tokenizer:
            if model_name_or_path is None:
                loading_path = args.tokenizer_name
                logger.info(f"Loading default tokenizer {loading_path}")
            else:
                loading_path = model_name_or_path
            tokenizer = token_cls.from_pretrained(loading_path)

        if not config:
            if model_name_or_path is None:
                logger.info(f"Loading config from args")
                config = config_cls(**args.model_config)
                config = self._init_vocab_size(config)
            else:
                config = config_cls.from_pretrained(model_name_or_path)
        
        config.batch_size = args.train_micro_batch_size_per_gpu
        config.max_seq_length = 512
        self.args.vocab_size = config.vocab_size

        self.tokenizer = tokenizer
        self.config = config
        self.config.num_hidden_groups = args.model_config["num_hidden_groups"]
        if "albert" not in model_type:
            self.network = model_cls(self.config, self.args, **model_kwargs) # bert
        else:
            self.network = model_cls(self.config, self.args) # albert
        # self.network = model_cls.from_pretrained(model_name_or_path)
        print(f"Before sharing total Parameter Count: {self.network.num_parameters()/1000/1000}M")

        # checkpoint_state_dict = torch.load("/mnt/liupeiyu/checkpoint/albert-base-v2/pytorch_model.bin", map_location=lambda storage, loc: storage)
        # moduel = checkpoint_state_dict['module'] if hasattr(checkpoint_state_dict, "module") else checkpoint_state_dict
        # self.network.load_state_dict(moduel, strict=False)
        if "albert" not in model_type and args.model_config["mpo_layers"] and args.model_config["mpo_layers"].lower() != "nompo":
            print("========== MPO Operation ==========")
            model = model_from_pretrained_mpo(self, args)
            if args.model_config["share_layer"]:
                args.model_config["share_layer"] = list(map(int, args.model_config["share_layer"].split(",")))
                if args.model_config["share_layer"]:
                    layer_idices = {args.model_config["share_layer"][0]:list(range(*args.model_config["share_layer"]))}
                else:
                    layer_idices = {0:list(range(1,args.model_config["num_hidden_layers"]))}
            else:
                layer_idices = {}
            logger.info("Check sharing config: {}".format(layer_idices))
            if layer_idices:
                names_tobe_shared = chain_module_names("bert", layer_idices=layer_idices)
                model.network.set_names_tobe_shared(names_tobe_shared)
                model.network.replace(root_name="")
                del model.network.names_tobe_shared
            else:
                print(f"No parameter sharing")
        elif "albert" in model_type and args.model_config["mpo_layers"] and args.model_config["mpo_layers"].lower() != "nompo" and args.model_config["num_hidden_groups"] > 1:
            print("========== ALBERT MPO sharing ==========")
            if args.model_config["load_from_albert"]:
                logger.info("===== Check loading weights from {} to ALBERT =====".format(args.model_config["load_from_albert"]))
                checkpoint_state_dict = torch.load(args.model_config["load_from_albert"], map_location=lambda storage, loc: storage)
                module = checkpoint_state_dict['module'] if hasattr(checkpoint_state_dict, "module") else checkpoint_state_dict
                self.network.load_state_dict(module, strict=False)

                del module
            # ------- share ALBERT weights
            # if args.model_config["share_layer"]:
            #     args.model_config["share_layer"] = list(map(int, args.model_config["share_layer"].split(",")))
            #     if args.model_config["share_layer"]:
            #         layer_idices = {args.model_config["share_layer"][0]:list(range(*args.model_config["share_layer"]))}
            #     else:
            #         layer_idices = {0:list(range(1,args.model_config["num_hidden_layers"]))}
            # else:
            #     layer_idices = {}
            # if layer_idices:
            #     names_tobe_shared = chain_module_names_orialbert("albert", layer_idices=layer_idices)
            #     self.network.set_names_tobe_shared(names_tobe_shared)
            #     self.network.replace(root_name="",verbose=False)
            #     del self.network.names_tobe_shared
            # else:
            #     print(f"No parameter sharing")
            # ------- MPO decom and share CT
            self = albert_from_pretrained_mpo(self, args)
            if args.model_config["share_layer"]:
                # args.model_config["share_layer"] = list(map(int, args.model_config["share_layer"].split(",")))
                if args.model_config["share_layer"] == '2CT':
                    layer_idices = {0:[1,2,3,4,5],6:[7,8,9,10,11]}
                    logger.info("Check sharing: {}".format(layer_idices))
                    # layer_idices = {args.model_config["share_layer"][0]:list(range(*args.model_config["share_layer"]))}
                else:
                    layer_idices = {0:list(range(1,args.model_config["num_hidden_layers"]))}
            else:
                layer_idices = {}
            if layer_idices:
                names_tobe_shared = chain_module_names_mpoalbert("albert", layer_idices=layer_idices)
                self.network.set_names_tobe_shared(names_tobe_shared)
                self.network.replace(root_name="")
                del self.network.names_tobe_shared
            else:
                print(f"No parameter sharing")
        ########### 原始的albert全做MPO分解 group=1
        elif "albert" in model_type and args.model_config["mpo_layers"] and args.model_config["mpo_layers"].lower() != "nompo" and args.model_config["num_hidden_groups"] == 1:
            print("========== test only mpo + ALBERT ==========")
            checkpoint_state_dict = torch.load(args.model_config["load_from_albert"], map_location=lambda storage, loc: storage)
            module = checkpoint_state_dict['module'] if hasattr(checkpoint_state_dict, "module") else checkpoint_state_dict
            self.network.load_state_dict(module, strict=False)
            self = albert_from_pretrained_mpo(self, args) # 测试MPO albert
            del module
        ########### albert 全共享
        else: 
            print("========== ALBERT ori sharing ==========")
            checkpoint_state_dict = torch.load(args.model_config["load_from_albert"], map_location=lambda storage, loc: storage)
            module = checkpoint_state_dict['module'] if hasattr(checkpoint_state_dict, "module") else checkpoint_state_dict
            self.network.load_state_dict(module, strict=False)
            del module
            if args.model_config["share_layer"]:
                args.model_config["share_layer"] = list(map(int, args.model_config["share_layer"].split(",")))
                if args.model_config["share_layer"]:
                    layer_idices = {args.model_config["share_layer"][0]:list(range(*args.model_config["share_layer"]))}
                else:
                    layer_idices = {0:list(range(1,args.model_config["num_hidden_layers"]))}
            else:
                layer_idices = {}
            logger.info("Check sharing config: {}".format(layer_idices))
            if layer_idices:
                names_tobe_shared = chain_module_names_orialbert("albert", layer_idices=layer_idices)
                self.network.set_names_tobe_shared(names_tobe_shared)
                self.network.replace(root_name="")
                del self.network.names_tobe_shared
            else:
                print(f"No parameter sharing")
        # for k,v in self.network.named_parameters():
        #     logger.info("{} nums: {}".format(k, v.numel()/1000/1000))
        # logger.info(self.network)
        # logger.info(["{}: {}\n".format(k,v.flatten()[:5]) for k,v in self.network.named_parameters()])
        print(f"After sharing total Parameter Count: {self.network.num_parameters()/1000/1000}M")
    def forward(self, batch):
        # outputs = self.network(batch)
        # ======= for albert
        input_ids = batch[1]
        token_type_ids = batch[3]
        attention_mask = batch[2]
        masked_lm_labels = batch[5]
        next_sentence_label = batch[4]
        total_loss = self.network.forward(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=masked_lm_labels,
                                sentence_order_label=next_sentence_label)
        return total_loss.loss
        # ======= for BERT
        # outputs = self.network.module(batch)
        # return outputs[0]  # return train=loss or infer=prediction scores

    @staticmethod
    def _init_vocab_size(config):
        # Padding for divisibility by 8
        if config.vocab_size % 8 != 0:
            config.vocab_size += 8 - (config.vocab_size % 8)
        logger.info(f"VOCAB SIZE: {config.vocab_size}")
        return config

    def save_weights(self, checkpoint_id, output_dir, is_deepspeed=False) -> str:
        """Save model weights, config and tokenizer configurations + extra arguments"""
        checkpoint_dir = os.path.join(output_dir, checkpoint_id)
        logger.info("checkpointing: PATH={}".format(checkpoint_dir))
        os.makedirs(checkpoint_dir, exist_ok=True)

        if is_deepspeed:
            # deepspeed save method
            self.network.module.save_pretrained(checkpoint_dir)
            # save Deepspeed config and running args (for future use)
            ds_config_path = os.path.join(checkpoint_dir, "deepspeed_config.json")
            self.to_json_file(self.args.ds_config, ds_config_path)
            self.args.deepspeed_config = ds_config_path
        else:
            # non deepspeed saving method
            self.network.save_pretrained(checkpoint_dir)

        self.config.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        args_file_path = os.path.join(checkpoint_dir, "args.json")
        args_dict = to_sanitized_dict(self.args)
        self.to_json_file(args_dict, args_file_path)

        return checkpoint_dir

    @classmethod
    def to_json_file(cls, dict_object, json_file_path):
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(cls.to_json_string(dict_object))

    @staticmethod
    def to_json_string(dict_object):
        return json.dumps(dict_object, indent=2, sort_keys=True) + "\n"

    def eval(self):
        self.network.eval()

    def train(self):
        self.network.train()

    def prepare_optimizer_parameters(self, weight_decay, CT_lr=None):
        logger.info("Check CT lr = {}".format(CT_lr))
        param_optimizer = list(self.network.named_parameters())
        param_optimizer = [n for n in param_optimizer if "pooler" not in n[0]]
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        CT_names = [n for n, p in param_optimizer if "tensor_set.2" in n]
        if CT_lr > 0.0:
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and not n in CT_names],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
                {
                    "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and n in CT_names],
                    "lr":CT_lr,
                    "weight_decay": weight_decay,
                },
            ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                }
            ]
        assert sum([sum([j.numel() for n, j in param_optimizer])]) == sum([sum([j.numel() for j in i['params']]) for i in optimizer_grouped_parameters])
        return optimizer_grouped_parameters
