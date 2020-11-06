import numpy as np
from PIL import Image
import cv2
import math
import tensorflow as tf
from functools import partialmethod
from os.path import join, isdir
from os import listdir

from utils import PreprocessInput
from utils import DepreprocessInput
from viz import ShowImage

class MyDataset(object):
    '''
    dataset directory structure:
    --Train
    ----Images
    ------xxx.jpg (other format also supported)
    --Val
    ----Images
    ------xxx.jpg
    '''
    def __init__(self,
                 root = None,
                 batch_size = 16, 
                 ratio = 2,  # 2x SR, 4x SR, 8x Sr, etc.
                 interpolation = 'cubic',
                 number_batch_tr = None,
                 number_batch_val = None, 
                 ):
        
        self.batch_size = batch_size
        self.ratio = ratio
        
        assert interpolation in ['cubic', 'bilinear']
        if interpolation == 'cubic':
            self.interpolation = cv2.INTER_CUBIC
        elif interpolation == 'bilinear':
            self.interpolation = cv2.INTER_LINEAR
            
        assert isdir(root)
        folder_tr = join(root, 'Train', 'Images')
        folder_val = join(root, 'Val', 'Images')
        self.files_tr = [join(folder_tr, name) for name in listdir(folder_tr)]
        self.files_val = [join(folder_val, name) for name in listdir(folder_val)]
        self.size_files_tr = len(self.files_tr)
        self.size_files_val = len(self.files_val)
        
        self.number_batch_tr = number_batch_tr
        self.number_batch_val = number_batch_val
        
        self.TransTr = self.GetTransformFnTr()
        self.TransVal = self.GetTransformFnVal()
        
    def GetTransformFnTr(self):
        raise NotImplementedError
        
    def GetTransformFnVal(self):
        raise NotImplementedError
        
    def GetShapeLow(self):
        # get the shape of low resolution, for example, (512, 512, 3), (1024, 768, 1), ... etc
        raise NotImplementedError
        
    def GetShapeHigh(self):
        # get the shape of high resolution
        h, w, c = self.GetShapeLow()
        h = h * self.ratio
        w = w * self.ratio
        return (h, w, c)
        
    def _MakeDividable(self, number, ratio):
        # crop down an image so that its shape can be dividable by ratio
        return int(math.floor(number/ratio)*ratio)
    
    def _GetImageHR(self, path):
        # read high resolution image from path
        img = Image.open(path)
        # step(1): we ignore gray style images
        if img.mode == 'L': return None
        if img.mode != 'RGB': raise ValueError('image {} not recognized with mode {}'.format(path, img.mode))
        # step(2): we ignore small images which do not fit into a patch 
        w, h = img.size
        right = self._MakeDividable(w, self.ratio)
        bottom = self._MakeDividable(h, self.ratio)
        
        min_h, min_w, c = self.GetShapeHigh()
        if bottom < min_h: return None
        if right < min_w: return None
        
        img = img.crop((0, 0, right, bottom))
        return img
    
    def _GetImageLR(self, pil_img_hr):
        # get low resolution image from a high resolution image (PIL Image)
        img = np.array(pil_img_hr)
        h = img.shape[0] 
        w = img.shape[1]
        assert h % self.ratio == 0
        assert w % self.ratio == 0
        h_new = int(h/self.ratio)
        w_new = int(w/self.ratio)
        dim = (w_new, h_new)
        img = cv2.resize(img, dim, interpolation=self.interpolation)
        np.clip(img, 0, 255, out=img)
        img_lr = Image.fromarray(img)
        return img_lr
    
    @staticmethod
    def _IndexIterator(size):
        while True:
            permutation = np.random.permutation(range(size))
            for idx in range(size):    
                yield permutation[idx]
            
    def _GetGenerator(self, state):
        assert state in ['train', 'val']
        names = self.files_tr if state == 'train' else self.files_val
        T = self.TransTr if state == 'train' else self.TransVal
      
        #iterator
        idx_iterator = MyDataset._IndexIterator(len(names))
        while True:
            while True:
                idx = next(idx_iterator)
                path = names[idx]
                img_hr = self._GetImageHR(path)
                if img_hr is not None: break
                
            img_lr = self._GetImageLR(img_hr)
            # T converts PIL image to ndarray image
            img_lr, img_hr = T(img_lr, img_hr)

            img_lr = PreprocessInput(img_lr)
            img_hr = PreprocessInput(img_hr)
            yield (img_lr, img_hr)
            
    _GetG = partialmethod(_GetGenerator)
    
    def _FnGenTr(self):
        return self._GetG(state='train')
    
    def _FnGenVal(self):
        return self._GetG(state='val')
    
    def GetDataset(self, infinite=True):
        '''
        if infinite is true, the virtual datasize is infinite, the dataset 
        iterates forever. Set to True if you are using keras default training 
        API, otherwize the iteration will run out eventually. In user-defined 
        training process, it is convenient to set to False.
        '''

        shape_lr = self.GetShapeLow()
        shape_hr = self.GetShapeHigh()
        
        shape_input = (shape_lr, shape_hr)
        type_input = (tf.float32, tf.float32)
        # note that the first arg is not a generator, instead, this is a function which returns a generator
        ds_tr = tf.data.Dataset.from_generator(self._FnGenTr, type_input, shape_input)
        ds_val = tf.data.Dataset.from_generator(self._FnGenVal, type_input, shape_input)
        

        ds_tr = ds_tr.batch(self.batch_size)
        ds_val = ds_val.batch(self.batch_size)
        
        if self.number_batch_tr is None or self.number_batch_val is None:
            infinite = True
            
        if not infinite:
            # how many batches does an epoch has
            # epoch_batch_tr = self.size_epoch_tr // self.batch_size
            # epoch_batch_val = self.size_epoch_val // self.batch_size
            ds_tr = ds_tr.take(self.number_batch_tr)
            ds_val = ds_val.take(self.number_batch_val)
            
        # prefetch batches
        ds_tr = ds_tr.prefetch(20)
        ds_val = ds_val.prefetch(20)
        
        return ds_tr, ds_val
    
    def _Test(self):
        ds_tr, ds_val = self.GetDataset()
        for batch_lr, batch_sr in ds_tr.take(1):
            break
        
        batch_lr = DepreprocessInput(batch_lr.numpy())
        batch_sr = DepreprocessInput(batch_sr.numpy())
        for idx in range(self.batch_size):
            lr = batch_lr[idx, ...]
            sr = batch_sr[idx, ...]
            ShowImage(lr)
            ShowImage(sr)
        
        
        
