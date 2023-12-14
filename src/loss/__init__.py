import torch
import functools

from src.loss.loss import (CrossEntropyLoss)


key2loss = {
    "CrossEntropyLoss": CrossEntropyLoss,
}


def get_loss_function(cfg):
    loss_dict = cfg["loss"]
    loss_name = loss_dict["name"]
    if loss_name not in key2loss:
        raise NotImplementedError("Loss {} not implemented".format(loss_name))

    return key2loss[loss_name]()
