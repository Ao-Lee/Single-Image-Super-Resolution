import numpy as np
import numbers
import random
# import math
from PIL import Image
from torchvision.transforms import functional as F

class MyCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label
  
class ToNumpy(object): # convert PIL to numpy
    def __call__(self, image, label):
        return np.array(image), np.array(label)

class MyRandomHFlip(object):
    def __call__(self, img_lr, img_hr):
        if np.random.random() > 0.5:
            return F.hflip(img_lr), F.hflip(img_hr)
        else:
            return img_lr, img_hr
    
class MyRandomCrop(object):
    def __init__(self, size, ratio):
        self.ratio = ratio
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
            
    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img_lr, img_hr):
        i, j, h, w = self.get_params(img_lr, self.size)
        return F.crop(img_lr, i, j, h, w), F.crop(img_hr, i*self.ratio, j*self.ratio, h*self.ratio, w*self.ratio)



