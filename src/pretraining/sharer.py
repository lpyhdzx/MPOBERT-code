# -*- coding: utf-8 -*-
# Date: 2022.03.22
# @author: LiuPeiyu
# @emal: liupeiyustu@163.com
import logging
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pretraining.compress_tools_v2.MPOtorch import LinearDecomMPO

_bert_roberta_names = {
    "K": lambda ptl, l: f"{ptl}.encoder.layer.{l}.attention.self.key_mpo",
    "Q": lambda ptl, l: f"{ptl}.encoder.layer.{l}.attention.self.query_mpo",
    "V": lambda ptl, l: f"{ptl}.encoder.layer.{l}.attention.self.value_mpo",
    "AO": lambda ptl, l: f"{ptl}.encoder.layer.{l}.attention.output.dense_mpo",
    "I": lambda ptl, l: f"{ptl}.encoder.layer.{l}.intermediate.dense_mpo",
    "O": lambda ptl, l: f"{ptl}.encoder.layer.{l}.output.dense_mpo",
    "P": lambda ptl, l: f"{ptl}.pooler.dense",
}
_albert_roberta_names = {
    "K": f"albert.encoder.albert_layer_groups.0.albert_layers.0.attention.key",
    "Q": f"albert.encoder.albert_layer_groups.0.albert_layers.0.attention.query",
    "V": f"albert.encoder.albert_layer_groups.0.albert_layers.0.attention.value",
    "AO": f"albert.encoder.albert_layer_groups.0.albert_layers.0.attention.dense",
    "I": f"albert.encoder.albert_layer_groups.0.albert_layers.0.ffn",
    "O": f"albert.encoder.albert_layer_groups.0.albert_layers.0.ffn_output",
    "P": f"albert.pooler",
}

_albert_mpo_names = {
    "K": lambda ptl, l: f"{ptl}.encoder.albert_layer_groups.{l}.albert_layers.0.attention.key_mpo",
    "Q": lambda ptl, l: f"{ptl}.encoder.albert_layer_groups.{l}.albert_layers.0.attention.query_mpo",
    "V": lambda ptl, l: f"{ptl}.encoder.albert_layer_groups.{l}.albert_layers.0.attention.value_mpo",
    "AO": lambda ptl, l: f"{ptl}.encoder.albert_layer_groups.{l}.albert_layers.0.attention.dense_mpo",
    "I": lambda ptl, l: f"{ptl}.encoder.albert_layer_groups.{l}.albert_layers.0.ffn_mpo",
    "O": lambda ptl, l: f"{ptl}.encoder.albert_layer_groups.{l}.albert_layers.0.ffn_output_mpo",
    "ILN": lambda ptl, l: f"{ptl}.encoder.albert_layer_groups.{l}.albert_layers.0.full_layer_layer_norm",
    "ALN": lambda ptl, l: f"{ptl}.encoder.albert_layer_groups.{l}.albert_layers.0.attention.LayerNorm",
    "P": lambda ptl, l: f"{ptl}.pooler",
}

_albert_ori_names = {
    "K": lambda ptl, l: f"{ptl}.encoder.albert_layer_groups.{l}.albert_layers.0.attention.key",
    "Q": lambda ptl, l: f"{ptl}.encoder.albert_layer_groups.{l}.albert_layers.0.attention.query",
    "V": lambda ptl, l: f"{ptl}.encoder.albert_layer_groups.{l}.albert_layers.0.attention.value",
    "AO": lambda ptl, l: f"{ptl}.encoder.albert_layer_groups.{l}.albert_layers.0.attention.dense",
    "I": lambda ptl, l: f"{ptl}.encoder.albert_layer_groups.{l}.albert_layers.0.ffn",
    "O": lambda ptl, l: f"{ptl}.encoder.albert_layer_groups.{l}.albert_layers.0.ffn_output",
    "ILN": lambda ptl, l: f"{ptl}.encoder.albert_layer_groups.{l}.albert_layers.0.full_layer_layer_norm",
    "ALN": lambda ptl, l: f"{ptl}.encoder.albert_layer_groups.{l}.albert_layers.0.attention.LayerNorm",
    "P": lambda ptl, l: f"{ptl}.pooler",
}

logger =  logging.getLogger(__name__)

def chain_module_names(which_ptl, layer_idices, abbres=["K","Q","V","AO","I","O"]):
    _plt_names = _bert_roberta_names
    for layer_init, layer_follow in layer_idices.items():
        names_tobe_shared = {}
        for abbre in abbres:
            name_init = _plt_names[abbre](which_ptl, layer_init)
            for l in layer_follow:
                names_tobe_shared.update({_plt_names[abbre](which_ptl, l) : name_init})
    return names_tobe_shared

def chain_module_names_orialbert(which_ptl, layer_idices, abbres=["K","Q","V","AO","I","O","ILN","ALN"]):
    abbres=["K","Q","V","AO","I","O","ILN","ALN"]
    _plt_names = _albert_ori_names
    for layer_init, layer_follow in layer_idices.items():
        names_tobe_shared = {}
        for abbre in abbres:
            name_init = _plt_names[abbre](which_ptl, layer_init)
            for l in layer_follow:
                names_tobe_shared.update({_plt_names[abbre](which_ptl, l) : name_init})
    return names_tobe_shared

def chain_module_names_mpoalbert(which_ptl, layer_idices, abbres=["K","Q","V","AO","I","O","ILN","ALN"]):
    abbres=["K","Q","V","AO","I","O","ILN","ALN"]
    _plt_names = _albert_mpo_names
    names_tobe_shared = {}
    for layer_init, layer_follow in layer_idices.items():
        for abbre in abbres:
            name_init = _plt_names[abbre](which_ptl, layer_init)
            for l in layer_follow:
                names_tobe_shared.update({_plt_names[abbre](which_ptl, l) : name_init})
    return names_tobe_shared

def route_name_to_module(name):
    '''
    name = "bert.encoder.layer.0.attention"
    return "bert.encoder.layer[0].attention"
    '''
    # digit = re.search('\d+', name).group()
    digits = list(set(re.findall('\d+', name)))
    for digit in digits:
        name = name.replace(".{}".format(digit), "[{}]".format(digit))
    return name
class Sharer(object):
    def __init__(self, names_tobe_shared, model) -> None:
        self.names_tobe_shared = names_tobe_shared
        if hasattr(model, "network"): # Pretraining model.network, Fine-tuning model
            self.model = model.network # delete after sharing
        else:
            self.model = model

    def replace(self, model, root_name):
        if hasattr(model, "network"):
            model = model.network
        elif hasattr(model, "module"):
            model = model.module
        for attr_str in dir(model):
            target_attr = getattr(model, attr_str)
            if isinstance(target_attr, nn.Module):
                name = root_name + "." + attr_str if len(root_name) > 0 else attr_str
            if type(target_attr) == LinearDecomMPO:
                shared = False

                if name in self.names_tobe_shared:
                    mid_ind = int(len(target_attr.tensor_set) // 2)
                    shared = True
                    init_attr = eval("self.model."+route_name_to_module(self.names_tobe_shared[name]))
                    target_attr.tensor_set[mid_ind] = init_attr.tensor_set[mid_ind]
                    logger.info("\t {} tensor_set[{}] is shared with {}".format(name, mid_ind, self.names_tobe_shared[name]))
                if not shared:
                    logger.info("\t {} is not shared".format(name))
            elif type(target_attr) == nn.Linear:
                shared = False

                if name in self.names_tobe_shared:
                    # ALBERT
                    shared = True
                    init_attr = eval("self.model."+route_name_to_module(self.names_tobe_shared[name]))
                    target_attr = init_attr
                    logger.info("\t ALBERT {} weight is shared with {}".format(name, self.names_tobe_shared[name]))
                if not shared:
                    logger.info("\t {} is not shared".format(name))
        
        for sub_modules_name, sub_modules in model.named_children():
            self.replace(
                model=sub_modules,
                root_name=root_name + "." + sub_modules_name
                if len(root_name) > 0
                else sub_modules_name
            )
