from os.path import isfile

from .generated_utils import *
from .generated_config import *
import os
import numpy as np
import cv2
from torch.utils.data import Dataset
import time
import psutil
from sys import getsizeof
import torch
from utils.auxilary_functions import image_resize

class GeneratedLoader(Dataset):

    def __init__(self, nr_of_channels=1, fixed_size=(128, None)):

        self.fixed_size = fixed_size

        save_file = dataset_path + '/' + 'generated_1' + '.pt'

        if isfile(save_file) is False:
            data = []
            i = 0
            for image_name in data_image_names:
                print('Imagename: ' + image_name)
                t0 = time.time()
                if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                    if nr_of_channels == 1:  # Gray scale image -> MR image
                        image = cv2.normalize(cv2.imread(os.path.join(data_path, image_name)), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        image = image[:, :, np.newaxis]
                    else:  # RGB image -> street view
                        image = cv2.normalize(cv2.open(os.path.join(data_path, image_name)), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    #t1 = time.time()
                    #image = normalize_array(image)
                    t2 = time.time()
                    image = getRandomCrop(image, image_name)
                    t3 = time.time()
                    word_array, info_array = cropWords(image, image_name.rsplit('.')[0])
                    t4 = time.time()
                    for i in range(0, len(word_array)):
                        data.append([word_array[i].copy(), info_array[i]])
                    t5 = time.time()

                i +=1

                if i % (len(data_image_names)/10) == 0:
                    print(str(i) + '/' + str(data_image_names))
            torch.save(data, save_file)
        else:
            data = torch.load(save_file)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        img = self.data[index][0]
        transcr = self.data[index][1]


        # add augmentation
        # ?? elastic, morphology, resizing

        nheight = self.fixed_size[0]
        nwidth = self.fixed_size[1]
        if nheight is None:
            nheight = img.shape[0]
        if nwidth is None:
            nwidth = int(np.random.uniform(.8, 1.2) * img.shape[1] * nheight / img.shape[0])

        img = image_resize(img, height=nheight-16, width=nwidth)
       # img = centered(img, (nheight, int(1.2 * nwidth) + 32))
        img = torch.Tensor(img).float().unsqueeze(0)

        return img, transcr