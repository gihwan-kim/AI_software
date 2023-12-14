import copy
import torch
import torch.nn as nn

# from src.models.gpt2 import GPT2
from transformers import GPT2LMHeadModel, GPT2Config

from src.models.image2shape import Image2shape


def get_model(model_dict, type):
    if type == 'Text2color':
        arc = model_dict['arch']
        name = model_dict['name']

        hg_model = _get_huggingface_model(arc).from_pretrained(name)
        config = hg_model.config
        config.n_layer = model_dict['config']['n_layer']
        config.n_head = model_dict['config']['n_head']
        print(config)
        new_model = GPT2LMHeadModel(config=config)

        return new_model
    elif type == 'Image2shape' or type == 'Image2text':
        # center_crop = model_dict['center_crop']
        config = model_dict['config']
        model = _get_model_instance(type)
        model = model(config['clip_type'],
                      prefix_length=config['prefix_length'],
                      clip_length=config['prefix_length_clip'],
                      num_layers=config['num_layers'],
                      prefix_size=config['prefix_size'])
        print(model.preprocess)
        return model


def _get_huggingface_model(arch):
    try:
        return {
            'GPT2LMHeadModel': GPT2LMHeadModel,
        }[arch]
    except:
        raise ('Model {} not available'.format(arch))

def _get_model_instance(arch):
    try:
        return {
            'Image2shape': Image2shape,
            'Image2text': Image2shape
        }[arch]
    except:
        raise ('Model {} not available'.format(arch))
