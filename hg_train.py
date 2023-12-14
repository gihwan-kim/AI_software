import os
import oyaml as yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
from torch.nn.parallel.scatter_gather import gather
from torch.utils import data
from tqdm import tqdm

from src.dataset import get_dataset
from src.loss import get_loss_function
from src.models import get_model
from src.optimizer import get_optimizer

from src.models.transformer_utils import cp_gpt2_transformer_block_weights

# TODO - 11/22~
# GPT2 model 구현
# training loss graph 구현
# batch size 를 2 보다 크게할 경우 입력 크기가 모두 달라 batch 가 안됨

from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast


import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_tokenizer(name):
    return {
	'GPT2TokenizerFast': GPT2TokenizerFast,
    'PreTrainedTokenizerFast': PreTrainedTokenizerFast
    }[name]


def train(cfg):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

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

    # ## [INPUT EMBEDDING]
    # ## copy embeddings from huggingface to my gpt2
    # model.wte.load_state_dict( hg_model.transformer.wte.state_dict() )
    # model.wpe.load_state_dict( hg_model.transformer.wpe.state_dict() )

    # ## [OUTPUT EMBEDDING]
    # ## copy to output vocab
    # model.head.load_state_dict( hg_model.lm_head.state_dict() )

    # ## [TRANSFORMER BLOCK]
    # ## transformer block copy
    # model = cp_gpt2_transformer_block_weights(hg_model, model)

    model = hg_model.to(device)



    # Dataset, Data loader setting
    t_dataset = dataset(
                        get_tokenizer(tokenizer),
                        tokenizer_name,
                        hg_model.config,
                        True,
                        train_data_path,
                        **tokenizer_configs
                        )

    v_dataset = dataset(
                        get_tokenizer(tokenizer),
                        tokenizer_name,
                        hg_model.config,
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
    for param in model.parameters():
        param.requires_grad = True


    print(model)

    # optimizer
    optimizer = get_optimizer(cfg["training"], model)
    print(optimizer)
    epoch = 0
    while epoch <= cfg['training']['epoch']:
        total_loss = 0.0
        for idx, (inputs_ids, labels_ids, inputs, labels) in enumerate(t_loader):
            # [Training]
            # print(inputs_ids.shape)
            # print(labels_ids.shape)
            inputs_ids = torch.squeeze(inputs_ids, dim=1)
            labels_ids = torch.squeeze(labels_ids, dim=1)

            model.train()
            optimizer.zero_grad()
            # input_ids = inputs_ids.to(device)
            # labels_ids = labels_ids.to(device).to(torch.float32)

            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(inputs)
            # print(outputs.shape)
            outputs = torch.argmax(outputs, dim=2).to(torch.float32)
            loss = criterion(outputs, labels)
            loss.requires_grad = True

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # print(f'Epoch {epoch + 1}, Iteration {idx + 1}, Loss {loss.item()}')

            # [Validation]
            if (epoch+1) % cfg['training']['val_interval'] == 0:
                model.eval()
                with torch.no_grad():
                    for (x_val, label_val) in v_loader:
                        x_val = x_val.to(device)
                        label_val = label_val.to(device)
                        outputs = model(x_val)

         # 에폭이 끝날 때마다 평균 손실 출력
        print(f'Epoch {epoch + 1}, Average Loss: {total_loss / len(t_loader)}')
        epoch += 1

        state = {
            "iter": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer" : optimizer.state_dict(),
            }
        save_path = os.path.join(
                        "./runs",
                        "{}_{}_{}.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"], epoch),
                    )
        torch.save(state, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='config')
    parser.add_argument(
        '--config',
        nargs='?',
        type=str,
        default='configs/t2c.yml',
        help='Configuration file to use',
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    train(cfg)