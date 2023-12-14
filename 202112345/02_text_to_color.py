############## TEST LIB ##############
import os

import skimage.io as io
from PIL import Image

import torch
from transformers import PreTrainedTokenizerFast

from transformers import GPT2LMHeadModel, AutoConfig
from text2color_utils import generate_sample
############## TEST LIB ##############




'''
    [sample] Image to Shapes Code
'''
import pytorch_lightning as pl
from tqdm import trange
import pandas as pd
import os, sys
import time
import datetime
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__name__))))

# from transformer import Model # 본인 모델 라이브러리 import

def cli_main():
    pl.seed_everything(1234)
    os.makedirs('./release/shape/', exist_ok=True)

    ############### MODEL SETTINGS ###############
    model_path = './model/text2color/trainLoss0.07022798045012002_GPT2LMHeadModel_Text2ColorDataset_epoch52.ckpt'
    config = AutoConfig.from_pretrained('skt/kogpt2-base-v2')

    config.n_layer = 4
    config.n_head = 4
    model = GPT2LMHeadModel(config=config)
    tokenizer_config = {
                        'bos_token': '</s>',
                        'eos_token': '</s>',
                        'unk_token': '<unk>',
                        'pad_token': '<pad>',
                        'mask_token': '<mask>'
                        }

    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", tokenizer_config)
    sep_token = {'sep_token': '<|sep|>'}
    tokenizer.add_special_tokens(sep_token)
    sep_ids = tokenizer(sep_token['sep_token'], return_tensors="pt").input_ids

    model.resize_token_embeddings(len(tokenizer))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(f=model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.to(device)
    sep_ids = sep_ids.to(device)
    #############################################

    student_id = 201701981 # 본인의 학번
    task_list = ['image_to_shape', 'text_to_color']
    task = task_list[1] # [image_to_shape, text_to_color] 목적에 맞는 task 이름으로 설정

    # ---- Validation DATA에 대한 prediction --- #
    to_fn = f"./release/color/{student_id}.{task}.valid.txt"
    description = f"Generating Validation {task} {student_id}"
    excel = pd.read_excel("./dataset/valid/scene.all.xlsx", engine='openpyxl', index_col="id")

    ## Generating all outputs for the testing sentences
    start_id = 7000
    end_id = 8500
    start_time = time.time()
    with open(to_fn, 'w', encoding='utf-8') as f:
        # for id in trange(start=start_id, stop=end_id, desc=description):
        # for id in range(start_id, end_id):
        for id in trange(start_id, end_id, desc=description):
            text = excel['text'][id]
            # [B, tokens]
            input_ids = tokenizer(text, return_tensors='pt').input_ids
            input_ids = input_ids.to(device)

            # sep 삽입
            # contents = torch([input] + <sep>)
            context = torch.cat((input_ids, sep_ids), dim=1)
            completion = generate_sample(context=context, tokenizer=tokenizer, model=model, length=20, device=device)

            # post-processing
            output = ' , '.join(sorted(set(word for word in completion[len(text)-2:].split() if len(word) >= 3)))
            print(f"{id}\t{output}", file=f)
        f.close()
    end_time = time.time()

    '''
        # # ---- Test DATA에 대한 prediction ---
        to_fn = f"./release/shape/{student_id}.{task}.test.txt"
        description = f"Generating Test {task} {student_id}"
        excel = pd.read_excel("./release/test/scene.stu.xlsx", engine='openpyxl', index_col="id")
        ## Generating all outputs for the testing sentences
        start_id = 8500
        end_id = 10000
        start_time = time.time()
        with open(to_fn, 'w', encoding='utf-8') as f:
            for id in trange(start=start_id, stop=end_id, desc=description):
                # id 통한 데이터 불러오기, excel 활용
                # 데이터 전처리
                # 모델에 입력하고 출력
                # 출력 후처리 -> completion

                completion = ' '
                print(f"{id}\t{completion}", file=f)
                f.flush()
            f.close()
        end_time = time.time()
    '''

    print("[Save] Generated texts -- ", to_fn)
    # sec = (start_time - end_time)
    sec = (end_time - start_time)
    result = datetime.timedelta(seconds=sec)
    print(f"{task} {student_id} take: {result}")


if __name__ == '__main__':
    cli_main()