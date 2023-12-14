############## TEST LIB ##############
import os

import skimage.io as io
from PIL import Image

import torch
from transformers import GPT2Tokenizer
from image2sahpe_utils import Image2shape, generate_beam, generate2
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
    # model_path = './model/image2shape/dynamic_valLoss0.12103189747826279_clipcap_RN50x4_Image2ShapeDataset_epoch101.ckpt'

    # A
    # model_path = './model/image2shape/a_valLoss0.208010978228723_clipcap_RN50x4_Image2ShapeDataset_epoch40.ckpt'
    # model_path = './model/image2shape/dynamic_a_valLoss0.208010978228723_clipcap_RN50x4_Image2ShapeDataset_epoch40.ckpt'

    # B
    # model_path = './model/image2shape/b_valLoss0.1431636317347487_clipcap_RN50x4_Image2ShapeDataset_epoch70.ckpt'
    # model_path = './model/image2shape/dynamic_b_valLoss0.1431636317347487_clipcap_RN50x4_Image2ShapeDataset_epoch70.ckpt'

    # C
    # model_path = './model/image2shape/c_valLoss0.13407523614090558_clipcap_RN50x4_Image2ShapeDataset_epoch80.ckpt'
    model_path = './model/image2text/valLoss0.17149486023187638_clipcap_RN50x4_Image2ShapeDataset_epoch150.ckpt'
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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(f=model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=True)

    # print(checkpoint["model_state"]["clipcap_model"]["gpt"])
    model.to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    #############################################

    ############### PATH SETTINGS ###############
    image_dir = '/home/guest/gihwan/AI_software/term/202112345/dataset/valid/images'
    #############################################


    student_id = 201701981 # 본인의 학번
    task_list = ['image_to_shape', 'text_to_color', 'image_to_text']
    task = task_list[2] # [image_to_shape, text_to_color] 목적에 맞는 task 이름으로 설정

    # ---- Validation DATA에 대한 prediction --- #
    to_fn = f"./release/text/{student_id}.{task}.valid.txt"
    description = f"Generating Validation {task} {student_id}"
    excel = pd.read_excel("./dataset/valid/scene.all.xlsx", engine='openpyxl', index_col="id")

    ## Generating all outputs for the testing sentences
    start_id = 7000
    end_id = 8500
    start_time = time.time()
    with open(to_fn, 'w', encoding='utf-8') as f:
        for id in trange(start_id, end_id, desc=description):
        # for id in range(start_id, end_id):
            # id 통한 데이터 불러오기, excel 활용
            # 데이터 전처리
            # 모델에 입력하고 출력
            # 출력 후처리 -> completion

            file_name = excel['image_fn'][id]
            file_name = file_name.split('\\')[-1]
            filename = os.path.join(image_dir, file_name)
            image = io.imread(filename)
            image = Image.fromarray(image)

            image = model.preprocess(image).unsqueeze(0).to(device)
            # prefix = model.clip_model.encode_image(image).to(device, dtype=torch.qint8)
            prefix = model.clip_model.encode_image(image).to(device, dtype=torch.float32)
            prefix_embed = model.clipcap_model.clip_project(prefix).reshape(1, config['prefix_length'], -1)

            generated_text_prefix = generate_beam(model=model.clipcap_model,
                                                  tokenizer=tokenizer,
                                                  embed=prefix_embed,
                                                  beam_size=1,
                                                  entry_length=30)
            # generated_text_prefix = generate2(model=model.clipcap_model, tokenizer=tokenizer, embed=prefix_embed)
            # completion = sorted(set(generated_text_prefix[0].split()))
            # completion = ' , '.join(completion)
            print(f"{id}\t{generated_text_prefix[0]}", file=f)
            f.flush()
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