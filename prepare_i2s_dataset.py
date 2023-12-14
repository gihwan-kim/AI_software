import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse

import ast
import pandas as pd

from torchvision import transforms

def main(clip_model_type: str,
         type,
         data_path='/home/guest/gihwan/AI_software/dataset/train/scene.all.xlsx',):

    # 유일 값만 가져오는 함수
    # def load_data_df(train_data_path):
    #     df = pd.read_excel(train_data_path)
    #     df.drop('Unnamed: 0', axis=1, inplace=True)

    #     image_fn = df['image_fn'].copy()
    #     scene = df['scene'].copy()

    #     # extract colors at elements of scene.
    #     for idx, value in enumerate(scene):
    #         y_list = []
    #         # convert string to object
    #         parsed_data = ast.literal_eval(value)

    #         # extract color string
    #         for object in parsed_data:
    #             y_list.append(object[1][0])

    #         # get unique color
    #         y_set = sorted(set(y_list))

    #         # convert set to string
    #         y_string = ', '.join(str(x) for x in y_set)
    #         scene[idx] = y_string
    #     for idx, value in enumerate(image_fn):
    #         # ./data/train/images\0.jpg
    #         file_name = value.split('\\')
    #         image_fn[idx] = file_name[-1]
    #     return image_fn, scene

    def load_data_df(train_data_path):
        df = pd.read_excel(train_data_path)
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

    # CLIP load
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)

    # path setting
    images_root = f'/home/guest/gihwan/AI_software/dataset/{type}/images'
    out_path = f"/home/guest/gihwan/AI_software/dataset/{type}/clip_pkl_{clip_model_name}_{type}.pkl"
    data_path = f'/home/guest/gihwan/AI_software/dataset/{type}/scene.all.xlsx'

    # load image, caption data
    image_file_names, captions = load_data_df(data_path)
    print("%0d captions loaded from json " % len(captions))

    all_embeddings = []
    all_captions = []

    print(clip_model)
    print(preprocess)
    for i in tqdm(range(len(image_file_names))):
        filename = os.path.join(images_root, image_file_names[i])
        image = io.imread(filename)
        image = Image.fromarray(image)

        # 512x512x3 =preprocess=> [3, 224, 224]
        # preprocess(image) => torch.Size([3, 224, 224])
        # unsqueeze(0) => torch.Size([1, 3, 224, 224])
        image = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
        all_embeddings.append(prefix)
        all_captions.append(captions[i])
        if (i + 1) % 10000 == 0:
            print(i)
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--type', default=False, choices=('train', 'valid', 'test'))
    args = parser.parse_args()
    exit(main(args.clip_model_type, args.type))
