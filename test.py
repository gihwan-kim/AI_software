
import argparse
import oyaml as yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

from torch.utils.data import Dataset, DataLoader

from src.dataset import get_dataset
from src.loss import get_loss_function
from src.models import get_model
from src.optimizer import get_optimizer

from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast

import numpy as np

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_tokenizer(name):
    return {
	'GPT2TokenizerFast': GPT2TokenizerFast,
    'PreTrainedTokenizerFast': PreTrainedTokenizerFast
    }[name]


def test(cfg):


    train_data_path = cfg['data']['train_path']
    valid_data_path = cfg['data']['valid_path']

    tokenizer = cfg['data']['tokenizer']
    tokenizer_name = cfg['data']['tokenizer_name']
    tokenizer_configs = cfg['data']['tokenizer_configs']

    # print(tokenizer_configs)
    # print(type(tokenizer_configs))

    dataset = get_dataset(cfg['data']['dataset'])


    # loss fn
    criterion = get_loss_function(cfg['training'])

    # model
    model, hg_model = get_model(cfg['model'])

    # ------------
    # Model
    # ------------

    # if new tokens are added, we have to resize model.



    # Dataset, Data loader setting
    t_dataset = dataset(
                        get_tokenizer(tokenizer),
                        tokenizer_name,
                        hg_model.config,
                        cfg['testing']['is_test'],
                        True,
                        train_data_path,
                        **tokenizer_configs
                        )

    v_dataset = dataset(
                        get_tokenizer(tokenizer),
                        tokenizer_name,
                        hg_model.config,
                        cfg['testing']['is_test'],
                        True,
                        valid_data_path,
                        **tokenizer_configs
                        )

    t_loader = data.DataLoader(t_dataset,
                                batch_size=cfg['training']['batch_size'],
                                num_workers=cfg['training']['n_workers'],
                                )


    v_loader = data.DataLoader(v_dataset,
                                batch_size=cfg['validating']['batch_size'],
                                num_workers=cfg['validating']['n_workers'],
                                )

    tokenizer = v_dataset.tokenizer
    decoder_bos_id, decoder_eos_id = v_dataset.bos_token_id, v_dataset.eos_token_id

    model.resize_token_embeddings(len(t_dataset.tokenizer))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(f=cfg["validating"]["resume"], map_location=device)
    model.load_state_dict(checkpoint["model_state"])#, strict=False)
    model.to(device)
    print(model)
    ## ------------
    ## testing
    ## ------------
    import tqdm

    model.eval()

    with torch.no_grad():
        # for idx, (seps_indexs, inputs_ids, attentions_masks, y)  in enumerate(v_loader):
        for idx, (seps_indexs, inputs_ids, label, y)  in enumerate(v_loader):

            inputs_ids = torch.squeeze(inputs_ids, dim=1)
            # attentions_masks = torch.squeeze(attentions_masks, dim=1)
            inputs_ids = inputs_ids.to(device)
            # attentions_masks = attentions_masks.to(device)

            # # print("-----------------------------")
            # # print(model.generate(inputs_ids))
            # out = model.generate(
            #                 inputs_ids,top_k=50, top_p=0.95)
            # print(tokenizer.decode(out[0].to('cpu').tolist()))
            # # print("-----------------------------")

            ####################################################
            # prev = inputs_ids
            # output = inputs_ids
            # # tokens = np.array(inputs_ids)
            # for i in range(300):
            #     print(output.shape)
            #     outputs = model(output)
            #     logits = outputs[0]
            #     logits = logits[:, -1, :]

            #     log_probs = F.softmax(logits, dim=-1)

            #     _, prev = torch.topk(log_probs, k=1, dim=-1)

            #     output = torch.cat((output, prev), dim=1)

            # print(output.shape)
            # text = tokenizer.decode(output[0])
            # print(text)
            ####################################################


            outputs = model(inputs_ids)
            outputs = torch.argmax(outputs.logits, dim=2)
            outputs = outputs[0].to('cpu').tolist()
            label = torch.squeeze(label, dim=1)
            label = label[0].tolist()

            bos_idx = 0
            eos_idx = 0
            count = 0
            for o_idx, val in enumerate(outputs):
                if val == decoder_bos_id[0]:
                    if count == 0:
                        bos_idx = o_idx
                    count += 1
                if count == 2:
                    eos_idx = o_idx
                    break
            print(f'\tidx = {idx}\n\tresult = {tokenizer.decode(outputs[bos_idx+1:eos_idx])}')
            print(f'\tidx = {idx}\n\ty = {y}')
            print('========================================')
            exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='config')
    parser.add_argument(
        '--config',
        nargs='?',
        type=str,
        default='configs/test_t2c.yml',
        help='Configuration file to use',
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    test(cfg)