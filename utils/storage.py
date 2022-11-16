import csv
import os
import torch
import logging
import sys

import utils
from .other import device


def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def get_storage_dir():
    if "RL_STORAGE" in os.environ:
        return os.environ["RL_STORAGE"]
    return "storage"


def get_model_dir(model_name):
    return os.path.join(get_storage_dir(), model_name)


def get_status_path(model_dir, i=None):
    if i is None or i == 'NEW':
        mode = i
        unnumbered_path = os.path.join(model_dir, 'status.pt')
        if os.path.exists(unnumbered_path):
            return unnumbered_path
        i = 0
        while os.path.exists(os.path.join(model_dir, 'status_%i.pt'%i)):
            i += 1
        
        if mode is None:
            i -= 1
    
    return os.path.join(model_dir, "status_%i.pt"%i)


def get_status(model_dir, i=None):
    path = get_status_path(model_dir, i=i)
    data = torch.load(path, map_location=device)
    print('Loaded status from %s'%path)
    return data


def save_status(status, model_dir, i):
    path = get_status_path(model_dir, i)
    utils.create_folders_if_necessary(path)
    torch.save(status, path)


def get_vocab(model_dir):
    return get_status(model_dir)["vocab"]


def get_model_state(model_dir, i=None):
    return get_status(model_dir, i=i)["model_state"]


def get_txt_logger(model_dir):
    path = os.path.join(model_dir, "log.txt")
    utils.create_folders_if_necessary(path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(filename=path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger()


def get_csv_logger(model_dir):
    csv_path = os.path.join(model_dir, "log.csv")
    utils.create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)
