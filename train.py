import os
import oyaml as yaml
import time
import shutil
import sys

from torch.utils.tensorboard import SummaryWriter
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

from torch.nn import functional as nnf

# TODO - 11/22~
# GPT2 model 구현
# training loss graph 구현
# batch size 를 2 보다 크게할 경우 입력 크기가 모두 달라 batch 가 안됨

from transformers import GPT2Tokenizer, GPT2TokenizerFast, PreTrainedTokenizerFast

import os
import shutil
from datetime import datetime


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_tokenizer(name):
    return {
        'GPT2Tokenizer': GPT2Tokenizer,
	    'GPT2TokenizerFast': GPT2TokenizerFast,
        'PreTrainedTokenizerFast': PreTrainedTokenizerFast
    }[name]


def train(cfg, cfg_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_data_path = cfg['data']['train_path']
    valid_data_path = cfg['data']['valid_path']

    tokenizer = cfg['data']['tokenizer']
    tokenizer_name = cfg['data']['tokenizer_name']
    tokenizer_configs = cfg['data']['tokenizer_configs']

    dataset = get_dataset(cfg['data']['dataset'])

    # model
    model = get_model(cfg['model'],cfg['type'])

    epoch = 0
    # Resume from checkpoint
    # if cfg["training"]["resume"] is not None:
    #     if os.path.isfile(cfg["training"]["resume"]):
    #         ckpt = torch.load(cfg["training"]["resume"])
    #         model.load_state_dict(ckpt["model_state"])

    #         optimizer.load_state_dict(ckpt['optimizer'])
    #         #best_iou = ckpt['best_iou']
    #         epoch = ckpt['iter']
    model = model.to(device)

    if cfg['type'] == 'Text2color':
        third_parameter = model.config
        kwargs = tokenizer_configs
    elif cfg['type'] == 'Image2shape' or cfg['type'] == 'Image2text':
        third_parameter = model.preprocess
        kwargs = cfg['data']['dataset_config']

    # Dataset, Data loader setting
    t_dataset = dataset(
                        get_tokenizer(tokenizer),
                        tokenizer_name,
                        third_parameter,
                        'train',
                        True,
                        train_data_path,
                        **kwargs
                        )

    v_dataset = dataset(
                        get_tokenizer(tokenizer),
                        tokenizer_name,
                        third_parameter,
                        'valid',
                        True,
                        valid_data_path,
                        **kwargs
                        )

    t_loader = data.DataLoader(t_dataset,
                                batch_size=cfg['training']['batch_size'],
                                num_workers=cfg['training']['n_workers'],
                                )


    v_loader = data.DataLoader(v_dataset,
                                batch_size=cfg['validating']['batch_size'],
                                num_workers=cfg['validating']['n_workers'],
                                )

    # loss fn
    criterion = get_loss_function(cfg['training'])


    if cfg['type'] == 'Text2color':
        third_parameter = model.config

        # setting "ignore_index" to "pad_token"
        criterion.ignore_index = t_dataset.pad_token_id[0]

        # if new tokens are added, we have to resize model.
        model.resize_token_embeddings(len(t_dataset.tokenizer))
    else:
        criterion.ignore_index = 0

    # optimizer
    optimizer = get_optimizer(cfg["training"], model)

    ######## check point folder 생성 ########
    save_path_root = os.path.join(cfg['model']['root'], cfg['type'])
    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)

    # 현재 시간을 이용하여 랜덤한 숫자 생성
    current_time = datetime.now()
    random_folder_name = current_time.strftime("%Y%m%d%H%M%S%f")[:-3]  # 마이크로초까지 사용

    # 조합된 경로
    save_path_root = os.path.join(save_path_root, random_folder_name)

    # 폴더 생성
    os.makedirs(save_path_root)

    # config file 복사
    shutil.copy2(cfg_path, save_path_root)
    ######## check point folder 생성 ########

    print(model)
    print(save_path_root)

    # 모델 저장 위치에 tensorboard 도 같이 저장
    writer = SummaryWriter(log_dir=save_path_root)

    if cfg['type'] == 'Text2color':
        val_min_total_loss = sys.float_info.max
        train_min_total_loss = sys.float_info.max
        while epoch <= cfg['training']['epoch']:
            total_loss = 0.0
            for idx, (sep_idx, inputs_ids, mask) in tqdm(enumerate(t_loader)):
                # [Training]

                inputs_ids = torch.squeeze(inputs_ids, dim=1)
                mask = torch.squeeze(mask, dim=1)

                model.train()
                # print(model)
                optimizer.zero_grad()

                inputs_ids = inputs_ids.to(device)
                mask = mask.to(device)

                outputs = model(inputs_ids)
                for batch_i, logits in enumerate(outputs.logits):
                    logits = logits[sep_idx[batch_i]-1:-1]
                    labels = inputs_ids[batch_i][sep_idx[batch_i]:].flatten()
                    loss = nnf.cross_entropy(logits, labels, ignore_index=t_dataset.pad_token_id[0])
                    total_loss += loss.item()


                loss.backward()
                optimizer.step()

            # [Train backpropagation]
            cur_train_loss = total_loss / len(t_loader)
            writer.add_scalar("Loss/train", cur_train_loss, epoch)
            print(f'Epoch {epoch}, Average Loss: {cur_train_loss}')

            # 에폭이 끝날 때마다 평균 손실 출력
            if cur_train_loss < train_min_total_loss:
                train_min_epoch = epoch
                train_min_total_loss = cur_train_loss
                train_min_state = {
                    "iter": train_min_epoch,
                    "model_state": model.state_dict(),
                    "optimizer" : optimizer.state_dict(),
                    }
                if epoch % cfg['training']['train_interval'] == 0:
                    save_path = os.path.join(
                                    save_path_root,
                                    "trainLoss{}_{}_{}_epoch{}.ckpt".format(train_min_total_loss,
                                                                        cfg["model"]["arch"],
                                                                        cfg["data"]["dataset"],
                                                                        train_min_epoch))
                    torch.save(train_min_state, save_path)

            # [Validation]
            model.eval()
            with torch.no_grad():
                total_loss_val = 0.0
                for idx, (sep_idx, inputs_ids, mask) in tqdm(enumerate(v_loader)):
                    inputs_ids = torch.squeeze(inputs_ids, dim=1)
                    mask = torch.squeeze(mask, dim=1)
                    inputs_ids = inputs_ids.to(device)
                    mask = mask.to(device)

                    outputs = model(inputs_ids)
                    for batch_i, logits in enumerate(outputs.logits):
                        logits = logits[sep_idx[batch_i]-1:-1]
                        labels = inputs_ids[batch_i][sep_idx[batch_i]:].flatten()
                        loss = nnf.cross_entropy(logits, labels, ignore_index=t_dataset.pad_token_id[0])
                        total_loss_val += loss.item()

                cur_val_loss = total_loss_val / len(v_loader)
                if cur_val_loss < val_min_total_loss:
                    val_min_epoch = epoch
                    val_min_total_loss = cur_val_loss
                    print(f'Epoch {epoch}, MIN VALIDATION Average Loss: {cur_val_loss}')
                    val_min_state = {
                                "iter": val_min_epoch,
                                "model_state": model.state_dict(),
                                "optimizer" : optimizer.state_dict(),
                    }
                    if epoch % 10 == 0:
                        save_path = os.path.join(save_path_root,
                                                "valLoss{}_{}_{}_epoch{}.ckpt".format(val_min_total_loss,
                                                                                    cfg["model"]["arch"],
                                                                                    cfg["data"]["dataset"],
                                                                                    val_min_epoch))
                        torch.save(val_min_state, save_path)
            epoch += 1

                # # [batch, tokens, vocab]
                # for batch_i, logits in enumerate(outputs.logits):

                #     # [1]
                #     shifted_logits = logits.contiguous()
                #     shifted_label = labels[batch_i].contiguous()

                #     loss = criterion(shifted_logits, shifted_label)
                #     total_loss += loss.item()

                # loss.backward()
                # optimizer.step()

                # total_loss += loss.item()

                # [mini batch loss]
                # print(f'Epoch {epoch + 1}, Iteration {idx + 1}, Loss {loss.item()}')

                # [Validation]
                # if (epoch+1) % cfg['training']['val_interval'] == 0:
                #     model.eval()
                #     with torch.no_grad():
                #         for (x_val, label_val) in v_loader:
                #             x_val = x_val.to(device)
                #             label_val = label_val.to(device)
                #             outputs = model(x_val)

            # # 에폭이 끝날 때마다 평균 손실 출력
            # print(f'Epoch {epoch}, Average Loss: {total_loss / len(t_loader)}')
            # writer.add_scalar("Loss/train", total_loss / len(t_loader), epoch)
            # epoch += 1

            # state = {
            #     "iter": epoch + 1,
            #     "model_state": model.state_dict(),
            #     "optimizer" : optimizer.state_dict(),
            #     }
            # save_path = os.path.join(
            #                 "./runs",
            #                 "onlyinput_{}_{}_{}.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"], epoch),
            #             )
            # torch.save(state, save_path)
    elif cfg['type'] == 'Image2shape' or cfg['type'] == 'Image2text':
        val_min_total_loss = sys.float_info.max
        train_min_total_loss = sys.float_info.max
        while epoch <= cfg['training']['epoch']:
            total_loss = 0.0
            for idx, (mask, prefix, tokens) in tqdm(enumerate(t_loader)):
                # [Training]
                model.train()
                model.zero_grad()
                optimizer.zero_grad()
                tokens = tokens.to(device)
                mask = mask.to(device)
                prefix = prefix.to(device, dtype=torch.float32)

                # forward
                outputs = model(mask, prefix, tokens)
                logits = outputs.logits[:, t_dataset.prefix_length - 1: -1]

                loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(),
                                         ignore_index=0)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # [mini batch loss]
                # print(f'Epoch {epoch + 1}, Iteration {idx + 1}, Loss {loss.item()}')

            # [Train backpropagation]
            cur_train_loss = total_loss / len(t_loader)
            writer.add_scalar("Loss/train", cur_train_loss, epoch)
            print(f'Epoch {epoch}, Average Loss: {cur_train_loss}')

            # 에폭이 끝날 때마다 평균 손실 출력
            if cur_train_loss < train_min_total_loss:
                train_min_epoch = epoch
                train_min_total_loss = cur_train_loss
                train_min_state = {
                    "iter": train_min_epoch,
                    "model_state": model.state_dict(),
                    "optimizer" : optimizer.state_dict(),
                    }
                if epoch % cfg['training']['train_interval'] == 0:
                    save_path = os.path.join(
                                    save_path_root,
                                    "trainLoss{}_{}_{}_{}_epoch{}.ckpt".format(train_min_total_loss,
                                                                        cfg["model"]["arch"],
                                                                        cfg["model"]["config"]["clip_type"],
                                                                        cfg["data"]["dataset"],
                                                                        train_min_epoch))
                    torch.save(train_min_state, save_path)

            # [Validation]
            model.eval()
            with torch.no_grad():
                total_loss_val = 0.0
                for idx, (mask_val, prefix_val, tokens_val) in tqdm(enumerate(v_loader)):
                    tokens_val = tokens_val.to(device)
                    mask_val = mask_val.to(device)
                    prefix_val = prefix_val.to(device, dtype=torch.float32)

                    # forward
                    outputs_val = model(mask_val, prefix_val, tokens_val)
                    logits_val = outputs_val.logits[:, t_dataset.prefix_length - 1: -1]
                    loss_val = nnf.cross_entropy(logits_val.reshape(-1, logits_val.shape[-1]), tokens_val.flatten(),
                                                    ignore_index=0)
                    total_loss_val += loss_val.item()

                cur_val_loss = total_loss_val / len(v_loader)
                if cur_val_loss < val_min_total_loss:
                    val_min_epoch = epoch
                    val_min_total_loss = cur_val_loss
                    print(f'Epoch {epoch}, MIN VALIDATION Average Loss: {cur_val_loss}')
                    val_min_state = {
                                "iter": val_min_epoch,
                                "model_state": model.state_dict(),
                                "optimizer" : optimizer.state_dict(),
                    }
                    if epoch % 10 == 0:
                        save_path = os.path.join(save_path_root,
                                                "valLoss{}_{}_{}_{}_epoch{}.ckpt".format(val_min_total_loss,
                                                                                    cfg["model"]["arch"],
                                                                                    cfg["model"]["config"]["clip_type"],
                                                                                    cfg["data"]["dataset"],
                                                                                    val_min_epoch))
                        torch.save(val_min_state, save_path)
            epoch += 1


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
    config_path = args.config
    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    # print(cfg['training']['resume'])

    train(cfg, config_path)
