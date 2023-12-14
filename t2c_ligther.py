import torch
import torch.nn as nn
import torch.quantization as quant

import torch
from transformers import GPT2Tokenizer
# from src.models.image2shape import Image2shape

from transformers import PreTrainedTokenizerFast
from transformers import GPT2LMHeadModel, AutoConfig

import os
import oyaml as yaml
import argparse


def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
    os.remove('temp.p')
    return size


def main(args):
    tokenizer_config = {
                        'bos_token': '</s>',
                        'eos_token': '</s>',
                        'unk_token': '<unk>',
                        'pad_token': '<pad>',
                        'mask_token': '<mask>'
                        }

    config = AutoConfig.from_pretrained('skt/kogpt2-base-v2')

    config.n_layer = 4
    config.n_head = 4
    model = GPT2LMHeadModel(config=config)

    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", tokenizer_config)
    sep_token = {'sep_token': '<|sep|>'}
    tokenizer.add_special_tokens(sep_token)
    sep_ids = tokenizer(sep_token['sep_token'], return_tensors="pt").input_ids

    model.resize_token_embeddings(len(tokenizer))

    # print(model)
    model_path = args.path
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = 'cuda:0'
    checkpoint = torch.load(f=model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=True)

    ckpt_name =  os.path.basename(model_path)
    print(model)
    if args.type == 'dynamic':
        quant_model = torch.quantization.quantize_dynamic( model,
                                                          {nn.Conv1d, nn.Conv2d},
                                                          dtype=torch.qint8)
        # compare the sizes
        f=print_size_of_model(model,"full")

        # model:            fp32  	 Size (KB): 976680.297
        # clip_model:       fp32  	 Size (KB): 421317.639
        # clip_project:     fp32  	 Size (KB): 57528.98
        # gpt:              fp32  	 Size (KB): 497811.545

        print("===quantized===")
        # model:            int8  	 Size (KB): 911632.903
        # clip_model:       fp32  	 Size (KB): 360718.743
        # clip_project:     fp32  	 Size (KB): 14480.722
        # gpt:              fp32  	 Size (KB): 536409.939

        q=print_size_of_model(quant_model,"full")
        print("{0:.2f} times smaller".format(f/q))
    # elif args.type == 'static':
    #     model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    #     model_fused = torch.ao.quantization.fuse_modules(model, [['conv', 'relu', 'bn']])
    #     model_prepared = torch.ao.quantization.prepare(model_fused)

    #     # calibration
        state = {
            "model_state" : quant_model.state_dict()
        }
        torch.save(state, f'{args.type}_{ckpt_name}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='config')
    parser.add_argument(
        '--path',
        nargs='?',
        type=str
    )

    parser.add_argument(
        '--type',
        nargs='?',
        type=str
    )

    args = parser.parse_args()
    main(args)