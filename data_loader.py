#!/usr/bin/env python
import numpy as np
from torch.utils.data.dataset import Dataset
from random import shuffle

import util

# training data loader
class TrainingSetLoader(Dataset):
    def __init__(self, root, transforms=None):
        """ Args:
            root (string): path to img file folder
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.root = root
        self.images = self.retrieve_images()
        self.transforms = transforms

    def __getitem__(self, index):
        # get the images based on the index 
        item = self.images[index]
        # Return image matrix and the corresponding classification
        return item['mat'], item['val']

    def __len__(self):
        return len(self.images)

    # retrieve every image
    def retrieve_images(self):
        images,labels,_ = util.read_dataset(self.root)
        img_ph_list,img_bg_list = util.crop_out(images,labels)
        ph_dict = [{'mat': np.rollaxis(x, 2, 0) , 'val': 1} for x in img_ph_list]
        bg_dict = [{'mat': np.rollaxis(x, 2, 0) , 'val': 0} for x in img_bg_list]
        mix_dict = ph_dict + bg_dict
        shuffle(mix_dict)
        return mix_dict



