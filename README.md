### Training Dataset
 - data to pickle file


1. image to shape
    ```
    python prepare_i2s_dataset.py --clip_model_type RN50 --type train
    ```

2. text to color
    ```
    python prepare_i2t_dataset.py --clip_model_type RN50 --type train
    ```


### Training

```
nohup python -u train.py --config ./configs/train_t2c.yml > ./logs/{lofg_file_name}.txt &

nohup python -u train.py --config ./configs/train_i2s.yml > ./logs/{lofg_file_name}.txt &
```

### Quantization

1. i2s
- `python {type}_lighter.py --path {ckpt_model_paht} --type {dynamic}`

    ```
    python i2s_ligther.py --path /home/guest/gihwan/AI_software/term/runs/Image2shape/20231211105651815/valLoss0.12748355151002624_clipcap_RN50x4_Image2ShapeDataset_epoch91.ckpt --type dynamic
    ```
