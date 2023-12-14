import json

from src.dataset.t2c_dataset import Text2ColorDataset
from src.dataset.i2s_dataset import Image2ShapeDataset
from src.dataset.i2t_dataset import Image2TextDataset

def get_dataset(name):
    """get_loader

    :param name:
    """
    return {
        "Text2ColorDataset": Text2ColorDataset,
        "Image2ShapeDataset": Image2ShapeDataset,
        "Image2TextDataset": Image2TextDataset,
    }[name]
