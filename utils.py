import os
import numpy as np
from skimage.transform import resize
from functools import partial
from tensorflow.keras import callbacks as KC

from viz import PlotResults

def PreprocessInput(batch_image):   
	return (batch_image / 127.5) - 1

def DepreprocessInput(batch_image):
    return ((batch_image + 1) * 127.5).astype(np.uint8)

def GetSchedular(lr_base=0.001, epoch_total=200):
    def _Scheduler(epoch_current, lr_current, plan):
        return plan[epoch_current]
    current_epoch = np.arange(epoch_total)
    plan = lr_base * (1 - current_epoch / epoch_total) ** 0.9
    Schedular = partial(_Scheduler, plan=plan)
    return Schedular

class DisplayPlot(KC.Callback):
    def __init__(self, root, ds, max_plots=3, **kwargs):
        super().__init__(**kwargs)
        self.ds_tr, self.ds_val = ds.GetDataset()
        self.max_plots = max_plots
        self.root = root
        if not os.path.isdir(root): os.makedirs(root)

    def get_prediction(self, data):
        img_lr, img_hr = data
        img_fake = self.model.predict_on_batch(img_lr)
        
        img_lr = DepreprocessInput(img_lr.numpy())
        img_hr = DepreprocessInput(img_hr.numpy())
        img_fake = DepreprocessInput(img_fake)
        
        shape = img_hr.shape # (B, h, w, c)
        img_lr = resize(img_lr, output_shape=shape, preserve_range=True)
    
        list_imgs = [img_lr, img_fake, img_hr]
        return list_imgs
        
        
    def on_epoch_end(self, epoch, logs=None):

        for data in self.ds_tr.take(1):
            break
        
        list_imgs = self.get_prediction(data)
        list_names = ['Train: LR', 'Train: Pred', 'Train: HR']
        
        for data in self.ds_val.take(1):
            break
        
        list_imgs += self.get_prediction(data)
        list_names += ['Val: LR', 'Val: Pred', 'Val: HR']
        
        # show at most 3 images in a batch, discard the rest
        list_imgs = [imgs[:self.max_plots, ...] for imgs in list_imgs]
        
        #  getting the pixel values between [0, 1] to plot it
        list_imgs = [imgs/255 for imgs in list_imgs]
        file_name = 'Epoch{:05d}'.format(epoch) + '.png'
        file_path = os.path.join(self.root, file_name)
        PlotResults(list_imgs, list_names, file_path=file_path)
        