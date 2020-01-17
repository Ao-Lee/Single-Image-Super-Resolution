import numpy as np
import math
import cv2
from PIL import Image
from os.path import join
from os.path import isdir
from os import listdir

try:
    from .transforms import ToNumpy, MyCompose, MyRandomCrop
    from .utils import Preprocess
    
except ImportError:
    from transforms import ToNumpy, MyCompose, MyRandomCrop
    from utils import Preprocess
'''
dataset directory structure:
--Train
----Images
------xxx.jpg (other format also supported)
--Val
----Images
------xxx.jpg
'''
class MyDataset(object):
    def __init__(self,
                 root = None,
                 batch_size = 16,
                 ratio = 2,  # 2x SR, 4x SR, 8x Sr, etc.
                 num_channel = 3, # 3-channel training images
                 shape_lr = 64, # input size of low resolution image, can be integer or a list
                 interpolation = 'cubic',
                 debug = False,
                 ):
        
        assert isdir(root)
        self.debug = debug
        self.batch_size = 1 if self.debug else batch_size
        self.ratio = ratio
        self.num_channel = num_channel
        assert interpolation in ['cubic', 'bilinear']
        if interpolation == 'cubic':
            self.interpolation = cv2.INTER_CUBIC
        elif interpolation == 'bilinear':
            self.interpolation = cv2.INTER_LINEAR
        else:
            raise ValueError('interpolation method not understood')
            
        if isinstance(shape_lr, list):
            assert len(shape_lr == 2)
            self.shape_lr = tuple(shape_lr)
        elif isinstance(shape_lr, int):
            self.shape_lr = (shape_lr, shape_lr)
        else:
            raise ValueError('data type not understood')
        
        folder_tr = join(root, 'Train', 'Images')
        folder_val = join(root, 'Val', 'Images')
        self.files_tr = [join(folder_tr, name) for name in listdir(folder_tr)]
        self.files_val = [join(folder_val, name) for name in listdir(folder_val)]

        self.nums_train_files = len(self.files_tr)
        self.nums_val_files = len(self.files_val)
        
        self.iterator_tr = self._RandomIterator(self.nums_train_files)
        self.iterator_val = self._SequentialIterator(self.nums_val_files)
        self.transform = self.GetDefaultTransform()
    
    def GetDefaultTransform(self):
        operations = []
        if not self.debug:
            operations.append(MyRandomCrop(size=self.shape_lr, ratio=self.ratio))
        operations.append(ToNumpy())
        return MyCompose(operations)
        
    def T(self, img_lr, img_hr):
        return (img_lr, img_hr) if self.transform is None else self.transform(img_lr, img_hr)
      
    def Read(self, path):
        return Image.open(path)
    
    def _MakeDividable(self, number, ratio):
        '''
        cut down a number so that it can be dividable by ratio
        '''
        return int(math.floor(number/ratio)*ratio)
        
    def GetImageHighResolution(self, path):
        # read high resolution image from path

        img = self.Read(path)
        # step(1): we ignore gray style images
        if img.mode == 'L': return None # img = img.convert('RGB')
        if img.mode != 'RGB': raise ValueError('image {} not recognized with mode {}'.format(path, img.mode))
        # step(2): we ignore small images which do not fit into a patch 
        w, h = img.size
        if math.floor(h//self.ratio) < self.shape_lr[0]: return None
        if math.floor(w//self.ratio) < self.shape_lr[1]: return None
            
        right = self._MakeDividable(w, self.ratio)
        bottom = self._MakeDividable(h, self.ratio)
        img = img.crop((0, 0, right, bottom))
        return img
    
    def GetImageLowResolution(self, pil_img_hr):
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
    
    def GetShapeHighResolution(self):
        return tuple([ele*self.ratio for ele in self.shape_lr]) + (self.num_channel, )
        
    def GetShapeLowResolution(self):
        return self.shape_lr + (self.num_channel, )
        
    def PreprocessHr(self, img):
        return Preprocess(img)
    
    def PreprocessLr(self, img):
        return Preprocess(img)
        
    def GetGenerator_Tr(self):
        return self._GetGenerator(idx_iterator=self.iterator_tr,
                                  files=self.files_tr
                                  )
    
    def GetGenerator_Val(self):
        return self._GetGenerator(idx_iterator=self.iterator_val, 
                                  files=self.files_val
                                  )
         
    @staticmethod
    def _RandomIterator(size):
        while True:
            yield np.random.randint(low=0, high=size)
            
    @staticmethod
    def _SequentialIterator(size):
        while True:
            for idx in range(size):
                yield idx
                
    def _GetGenerator(self, idx_iterator, files):
        while True:
            batch_imgs_hr = []
            batch_imgs_lr = []
            for _ in range(self.batch_size):
                while True:
                    idx = next(idx_iterator)
                    file = files[idx]
                    img_hr = self.GetImageHighResolution(file)
                    if img_hr is not None: break
                img_lr = self.GetImageLowResolution(img_hr)
                img_lr, img_hr = self.T(img_lr, img_hr)
                batch_imgs_hr.append(img_hr)
                batch_imgs_lr.append(img_lr)

            batch_imgs_hr = np.array(batch_imgs_hr)
            batch_imgs_lr = np.array(batch_imgs_lr)
            batch_imgs_hr = self.PreprocessHr(batch_imgs_hr)
            batch_imgs_lr = self.PreprocessLr(batch_imgs_lr)
            yield (batch_imgs_lr, batch_imgs_hr)
