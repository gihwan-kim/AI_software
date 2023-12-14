
import torch
from transformers import GPT2Tokenizer
from torch.utils import data



import ast
import os
import sys

import pandas as pd
import cv2
from PIL import Image
import numpy as np

import pickle

# from transformers import GPT2TokenizerFast

# pkl 형태로 미리 준비하자
#   학습할 거는 mapping network 하나이기 때문에 굳이 clip embedding 을 학습하면서
#   만들 필요가 없다.

# # 한국어 GPT-2 모델의 토크나이저 불러오기
# tokenizer_korean = GPT2TokenizerFast.from_pretrained("klue/gpt2-base")
class Image2ShapeDataset(data.Dataset):
    def __init__(self,
                 tokenizer,
                 tokenizer_name, # gpt2
                 preprocess,
                 type='train',
                 is_all=True,
                 data_path='/home/guest/gihwan/AI_software/dataset/train/scene.all.xlsx',
                 **kwargs,):
        self.data_path = data_path
        self.images_root = ''

        # CLIP preprocess
        self.preprocess = preprocess
        self.tokenizer = tokenizer.from_pretrained(tokenizer_name)

        self.prefix_length = kwargs.get('prefix_length', None)
        self.normalize_prefix = kwargs.get('normalize_prefix', None)

        self.type = type

        self.images_root == f'/home/guest/gihwan/AI_software/dataset/{self.type}/images'
        self.data_pkl_path = kwargs.get(f'{self.type}_pkl_path', None)

        # all_data: clip_emgedding, captions
        with open(self.data_pkl_path, 'rb') as f:
             all_data = pickle.load(f)
        print("Data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()

        self.prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]     # list

        self.image_filenames, self.captions_text = self.load_data_df()
        self.captions = [caption for caption in captions_raw]

        self.captions_tokens = []
        max_seq_len = 0
        for caption in captions_raw:
            self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption), dtype=torch.int64))
            max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])

        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))


    def __len__(self):
        return len(self.captions_tokens)

    # read image
    # caption => token
    def __getitem__(self, idx):
        prefix = self.prefixes[idx]
        # if self.normalize_prefix:
        #     prefix = prefix.float()
        #     prefix = prefix / prefix.norm(2, -1)
        tokens, mask = self.pad_tokens(idx)

        return mask, prefix, tokens


    def load_data_df(self):
        df = pd.read_excel(self.data_path)
        df.drop('Unnamed: 0', axis=1, inplace=True)

        image_fn = df['image_fn'].copy()
        scene = df['scene'].copy()

        # extract colors at elements of scene.
        for idx, value in enumerate(scene):
            y_list = []
            # convert string to object
            parsed_data = ast.literal_eval(value)

            # extract color string
            for object in parsed_data:
                y_list.append(object[1][0])

            # convert set to string
            y_string = ' '.join(str(x) for x in y_list)
            scene[idx] = y_string.strip()
            scene[idx] = y_string

        for idx, value in enumerate(image_fn):
            # ./data/train/images\0.jpg
            file_name = value.split('\\')
            image_fn[idx] = file_name[-1]
        return image_fn, scene


    def load_data(self):
        data = []
        with open(self.dir, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()
                data.append(line)
        return data


    def pad_tokens(self, item: int):
        '''
            CLIP_prefix_caption
        '''
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask
