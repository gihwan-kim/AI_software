import torch
import torch.nn as nn
import torch.quantization as quant

import torch
from transformers import GPT2Tokenizer
from src.models.image2shape import Image2shape

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
    # config = {
    #                   'clip_type' : 'RN50x4',
    #                   'prefix_length' : 40,
    #                   'prefix_length_clip' :  40,
    #                   'num_layers' : 8,
    #                   'prefix_size' : 640
    #           }

    config = {
                      'clip_type' : 'RN50x4',
                      'prefix_length' : 10,
                      'prefix_length_clip' :  10,
                      'num_layers' : 2,
                      'prefix_size' : 640
              }
    model = Image2shape(config['clip_type'],
                        prefix_length=config['prefix_length'],
                        clip_length=config['prefix_length_clip'],
                        num_layers=config['num_layers'],
                        prefix_size=config['prefix_size'])

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
        clip_=print_size_of_model(model.clip_model,"clip")
        clip_cap=print_size_of_model(model.clipcap_model.clip_project,"proj")
        clip_project=print_size_of_model(model.clipcap_model.gpt,"gpt2")

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
        clip_=print_size_of_model(quant_model.clip_model,"clip")
        clip_cap=print_size_of_model(quant_model.clipcap_model.clip_project,"proj")
        clip_project=print_size_of_model(quant_model.clipcap_model.gpt,"gpt2")

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