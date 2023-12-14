import os
import torch
import numpy as np
import imageio as m
from transformers import GPT2Tokenizer
from torch.utils import data

import ast
import pandas as pd

class BaseDataset(data.Dataset):
    def __init__(self,
                 tokenizer,
                 tokenizer_name,
                 hg_model_config,
                 is_test=False,
                 is_all=True,
                 path='/home/guest/gihwan/AI_software/dataset/train/scene.all.xlsx',
                 **kwargs,):
        self.path = path

        if is_all:
            # scene.all.xlsx
            self.x, self.y = self.load_data_df()
        else:
            # file
            self.x, self.y = self.load_data()

        # Tokenizer setting
        self.tokenizer = tokenizer.from_pretrained(tokenizer_name,
                                                   kwargs)

        self.pad_token_id = self.tokenizer.encode('<pad>')
        self.bos_token_id = self.tokenizer.encode('</s>')
        self.eos_token_id = self.tokenizer.encode('</s>')
        self.mask_token_id = self.tokenizer.encode('<mask>')
        self.vocab_size = hg_model_config.vocab_size

        self.sep_token = '<|sep|>'
        self.special_tokens = {'sep_token': self.sep_token}

        self.tokenizer.add_special_tokens(self.special_tokens)

        self.is_test = is_test

    def __len__(self):
        return len(self.x)

    # batch size = 1
    def __getitem__(self, idx):

        text = self.x[idx]
        label_text = self.y[idx]
        y = label_text

        text = text.replace(';', '')

        # text => tokens
        input = self.tokenizer(text, return_tensors="pt")
        label = self.tokenizer(label_text, return_tensors="pt")
        sep = self.tokenizer(self.sep_token, return_tensors="pt")

        # input = torch( <bos> + [input] + <eos>)
        contents = add_bos_eos(input.input_ids, self.bos_token_id, self.eos_token_id)

        # target = torch( <bos> + [target] + <eos>)
        label_ids = add_bos_eos(label.input_ids, self.bos_token_id, self.eos_token_id)

        if self.is_test:
            return len(contents[0] - 1), contents, label_ids, y      # sep idx, input_ids, mask

        # training 은 batch size 를 맞춰주기 위해서 padding 을 넣어준다.
        contents = add_pad(contents, self.pad_token_id)
        label_ids = add_pad(label_ids, self.pad_token_id)
        return len(contents[0] - 1), contents, label_ids      # sep idx, input_ids, mask


    def load_data_all(self):
        df = pd.read_excel(self.path)
        df.drop('Unnamed: 0', axis=1, inplace=True)
        return df

    def load_data_df(self):
        df = pd.read_excel(self.path)
        df.drop('Unnamed: 0', axis=1, inplace=True)

        text = df['text'].copy()
        scene = df['scene'].copy()

        # extract colors at elements of scene.
        for idx, value in enumerate(scene):
            y_list = []
            # convert string to object
            parsed_data = ast.literal_eval(value)

            # extract color string
            for object in parsed_data:
                y_list.append(object[0])

            # get unique color
            y_set = sorted(set(y_list))

            # convert set to string
            y_string = ', '.join(str(x) for x in y_set)
            scene[idx] = y_string
        return text, scene

    # txt
    def load_data(self):
        data = []
        with open(self.dir, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()
                data.append(line)
        return data

    # vocab 내에서 해당 데이터가 있는지 확인하기
    def check_vocab(self, word_to_check):
        # 어휘 사전 확인
        vocab = self.tokenizer.get_vocab()

        # 어휘에 속하는지 여부 확인
        if word_to_check in vocab:
            print(f"{word_to_check} is in the vocabulary.")
        else:
            print(f"{word_to_check} is not in the vocabulary.")
            self.tokenizer.add_tokens(word_to_check)


def add_pad(x, pad_id, max_len=1024):
    _, currnet_len =  x.shape
    if currnet_len > max_len:
        return x[:, :max_len]

    padding_size = max_len - currnet_len
    padding_tensor = torch.full((x.size(0), padding_size), pad_id[0], dtype=x.dtype)

    # 시퀀스에 패딩 추가
    padded_sequence = torch.cat((x, padding_tensor), dim=1)

    return padded_sequence

def add_bos_eos(x, bos_id, eos_id):
    # [1, bos] concat [1, tokens] concat [1, eos]
    #   dim 1 에 대해 concat
    x = torch.cat((torch.tensor([bos_id]), x), dim=1)
    x = torch.cat((x, torch.tensor([eos_id])), dim=1)
    return x
