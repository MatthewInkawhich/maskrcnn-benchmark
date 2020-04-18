# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict
import logging

import torch

from maskrcnn_benchmark.utils.imports import import_file


def align_and_update_state_dicts(model_state_dict, loaded_state_dict, dont_load):
    """
    Strategy: suppose that the models that we will create will have prefixes appended
    to each of its keys, for example due to an extra level of nesting that the original
    pre-trained weights from ImageNet won't contain. For example, model.state_dict()
    might return backbone[0].body.res2.conv1.weight, while the pre-trained model contains
    res2.conv1.weight. We thus want to match both parameters together.
    For that, we look for each model weight, look among all loaded keys if there is one
    that is a suffix of the current weight name, and use it if that's the case.
    If multiple matches exist, take the one with longest size
    of the corresponding name. For example, for the same model as before, the pretrained
    weight file can contain both res2.conv1.weight, as well as conv1.weight. In this case,
    we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.
    """
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # loaded_key string, if it matches
    match_matrix = [
        len(j) if i.endswith(j) else 0 for i in current_keys for j in loaded_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(
        len(current_keys), len(loaded_keys)
    )
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    log_str_template = "{: <{}} loaded from {: <{}} of shape {}"
    logger = logging.getLogger(__name__)
    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            continue
        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        
        # Only load keys that do not contain the dont_load strings
        good_to_load = True
        for dl_string in dont_load:
            if dl_string in key:
                good_to_load = False
                break

        if good_to_load:
            model_state_dict[key] = loaded_state_dict[key_old]
            logger.info(
                log_str_template.format(
                    key,
                    max_size,
                    key_old,
                    max_size_loaded,
                    tuple(loaded_state_dict[key_old].shape),
                )
            )


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


# Use this to create a loadable state dict for the EWAdaptive model
def prep_for_ewadaptive_model(loaded_state_dict, branch_counts):
    new_state_dict = {}
    for k, v in loaded_state_dict.items():
        found = False
        for i in range(1, 4):
            if "layer"+str(i) in k:
                for b in range(branch_counts[i-1]):
                    new_key = k.replace("layer"+str(i), "C"+str(i+1)+".branch"+str(b))
                    new_state_dict[new_key] = v
                found = True
        if not found:
            new_state_dict[k] = v
    return new_state_dict


# Use this to add other branches to a primer model's state dict
def prep_primed_state_dict(loaded_state_dict, branch_counts):
    new_state_dict = {}
    for k, v in loaded_state_dict.items():
        new_state_dict[k] = v
        for i in range(2, 2+len(branch_counts)):
            if "C"+str(i) in k:
                for b in range(1, branch_counts[i-2]):
                    new_key = k.replace("branch0", "branch"+str(b))
                    new_state_dict[new_key] = v

    return new_state_dict


# Use this to format the pretrained resnet state_dict for the Strider backbone
def prep_for_strider(loaded_state_dict, strider_body_config):
    # Iterate thru keys, collecting into layer_prefixes
    layer_prefixes = []
    for k, _ in loaded_state_dict.items():
        if 'layer' in k:
            k_arr = k.split('.')
            prefix = k_arr[0] + '.' + k_arr[1]
            if prefix not in layer_prefixes:
                layer_prefixes.append(prefix)

    # Create layer --> block map
    layer_to_block_map = {}
    block_index = 0
    for layer_prefix in layer_prefixes:
        layer_to_block_map[layer_prefix] = "block" + str(block_index)
        block_index += 1

    # Create new_state_dict
    new_state_dict = {}
    for k, v in loaded_state_dict.items():
        for layer_prefix, block_prefix in layer_to_block_map.items():
            if layer_prefix in k:
                k = k.replace(layer_prefix, block_prefix)
                # Replace conv2.weight with conv2_weight if block is a StriderBlock
                block_idx = int(block_prefix.split('block')[-1])
                striderblock = strider_body_config[block_idx][0]
                if striderblock == 1:
                    k = k.replace("conv2.weight", "conv2_weight")
                break
        new_state_dict[k] = v 
    return new_state_dict


def load_state_dict(model, loaded_state_dict, dont_load=[], branch_counts=[], primer=False, strider_body_config=[]):
    # Create a record of the default model state dict so we can use strict loading later
    model_state_dict = model.state_dict()
    # if the state_dict comes from a model that was wrapped in a
    # DataParallel or DistributedDataParallel during serialization,
    # remove the "module" prefix before performing the matching
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")

    # If branch_counts arg is set
    if branch_counts:
        assert(len(branch_counts) == 3, "Error: length of branch_counts must == 3")
        if primer:
            loaded_state_dict = prep_primed_state_dict(loaded_state_dict, branch_counts)
        else:
            loaded_state_dict = prep_for_ewadaptive_model(loaded_state_dict, branch_counts)
    
    if strider_body_config:
        #print("\n\nBEFORE PREPPING:")
        #for k, v in loaded_state_dict.items():
        #    print(k)
        loaded_state_dict = prep_for_strider(loaded_state_dict, strider_body_config)
        #print("\n\nAFTER PREPPING:")
        #for k, v in loaded_state_dict.items():
        #    print(k)

    align_and_update_state_dicts(model_state_dict, loaded_state_dict, dont_load)

    # use strict loading
    model.load_state_dict(model_state_dict)
