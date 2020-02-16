import copy
import random
import numpy as np
from itertools import permutations
from tensorflow import keras
import time
from PIL import Image
import cv2
from imgaug import augmenters as iaa
import utils
import config

class BatchGenerator(keras.utils.Sequence):
    """
    This batch generator generates batches of augmented images, labels, and attributes.
    """
    def __init__(self,
                data_reader,
                batch_size = config.BATCH_SIZE_DEFAULT,
                input_size = config.INPUT_SIZE_DEFAULT,
                train_ratio = config.TRAIN_RATIO_DEFAULT,
                random_seed = config.RANDOM_SEED_DEFAULT,
                jitter = config.JITTER_DEFAULT,
                mode = 'train',
                # TODO Define arguments of your choice
                ):
        """
        :param data_reader: kitti data reader from reader.py
        :param batch_size:
        :param shuffle: true if shuffle dataset
        :param jitter: true if augment images
        :param mode: train or test
        """
        self.random_seed = random_seed
        random.seed(self.random_seed)
        assert mode in ['train', 'val']
        
        self.data_reader = data_reader
        self.batch_size = batch_size
        self.input_size = input_size
        self.aug_pipe = self.get_aug_pipeline(p = 0.5)
        self.index = list(range(100)) # TODO define index something like list(range(len(self.data_reader.image_data)))
        random.shuffle(self.index)
        self.mode = mode
        if self.mode == 'train' :
            self.index = self.index[:int(len(self.index) * train_ratio)]
        else:
            self.index = self.index[int(len(self.index) * train_ratio):]
        self.jitter = jitter
        
    def __len__(self):
        """
        :return :Number of batches in this generator
        """
        return int(len(self.index) / self.batch_size)

    def on_epoch_end(self):
        """
        Function called in the end of every epoch.
        """
        np.random.shuffle(self.index)

    def __get_bounds__(self, idx):
        """
        Retrieve bounds for specified index
        :param idx: index 
        :return left bound, right bound:
        """
        #Define bounds of the image range in current batch
        l_bound = idx*self.batch_size #left bound
        r_bound = (idx+1)*self.batch_size #right bound

        if r_bound > len(self.index):
            r_bound = len(self.index)
            # Keep batch size stable when length of images is not a multiple of batch size.
            l_bound = r_bound - self.batch_size
        return l_bound, r_bound

    def preprocess(self, image, annot):
        """
        Preprocessing function called in __getitem__
        :return: preprocessed image
        """
        # crop = aug_pipe.augment_image(crop)
        raise NotImplementedError()
        

    def get_aug_pipeline(self, p = 0.2):
        # Helper Lambda

        sometimes = lambda aug: iaa.Sometimes(p, aug)

        aug_pipe = iaa.Sequential(
            [
                iaa.OneOf(
                    [
                        sometimes(iaa.Multiply((0.5, 0.5), per_channel=0.5)),
                        sometimes(iaa.GaussianBlur((0, 3.0))),  # blur images with a sigma between 0 and 3.0
                        sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.04 * 255), per_channel=0.5)),
                        
                    ]
                )
            ],
            random_order=True
        )

        return aug_pipe  

    def __getitem__(self, i):
        """
        Abstract function from Sequence class - called every iteration in model.fit_generator function.
        :param i: batch id
        :return X, Y
        """
        X = []
        Y = []
        l_bound, r_bound = self.__get_bounds__(i)
        for j in range(l_bound, r_bound):
            idx = self.index[j]
            # TODO append to X and Y

        X = np.array(X)
        return X, Y

    def visualize(self, i):
        """
        Optional visualization function
        :param i: batch index
        :return: numpy image with visualized outputs
        """
        raise NotImplementedError()