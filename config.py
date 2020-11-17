import os
import json
import models
import tensorflow as tf

# Data path
DATA_DIR = os.path.abspath("data/training")

# training ratio
TRAIN_RATIO_DEFAULT = 0.8

# random seed used when splitting training and validation
RANDOM_SEED_DEFAULT = 10

# path to save model checkpoints and tensorboard logs
SAVE_TO_DEFAULT = "progress/test/"

# batch size
BATCH_SIZE_DEFAULT = 32

# number of epochs
NUM_EPOCHS_DEFAULT = 500

# image size (width, height)
INPUT_SIZE_DEFAULT = [224, 224]

# model name that matches a key models.MODELS
MODEL_DEFAULT = 'mobilenet_v1'

# apply image augmentations to images if true
JITTER_DEFAULT = True

# keep aspect ratio of cropped patches when resizing them to input size
KEEP_ASPECT_RATIO_DEFAULT = False

# MODELS dictionary that maps model names to construction functions
MODELS = {
    'mobilenet_v1': models.mobilenet_v1.construct,
}

# TODO define default variables of your choice


def save(
        save_to,
        model=None,
        input_size=None,
        jitter=None,
        batch_size=None,
        num_epochs=None,
        train_ratio=TRAIN_RATIO_DEFAULT,
        random_seed=RANDOM_SEED_DEFAULT,
        **kwargs):
    """
    Save configuration paramteres into file
    """
    path = os.path.join(save_to, 'config.json')
    assert model in MODELS.keys()
    data = {
        'save_to': save_to,
        'model': model,
        'train_ratio': train_ratio,
        'random_seed': random_seed,
        'jitter': jitter,
        'input_size': input_size,
        'batch_size': batch_size,
        'tf_version': tf.__version__
    }
    data.update(kwargs)

    with open(path, 'w') as f:
        json.dump(data, f)

    return data


def read(read_from):
    """
    Read configuration file
    :param read_from: directory where config.json file is
    :return: dictionary with config params
    """
    path = os.path.join(read_from, 'config.json')
    with open(path, 'r') as f:
        data = json.load(f)
    assert data['model'] in MODELS.keys()
    return data
