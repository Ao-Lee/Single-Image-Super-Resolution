import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from scipy.misc import imresize

def MergeImage(imgs, color=(40,40,40), interval=5):
    '''
    imgs has shape(num_row, num_col, dh, dw, 3)
    '''
    
    num_row, num_col, dh, dw, _ = imgs.shape
    total_h = num_row * (dh + interval) + interval
    total_w = num_col * (dw + interval) + interval
    result = np.zeros(shape=(total_h, total_w, 3), dtype=np.uint8)
    result[:,:,:] = np.array(color) 

    for ir in range(num_row):
        for ic in range(num_col):
            start_h = ir*(dh + interval) + interval
            start_w = ic*(dw + interval) + interval
            end_h = start_h + dh
            end_w = start_w + dw
            result[start_h:end_h, start_w:end_w, :] = imgs[ir, ic, :, :, :]

    return result


def ShowImage(img, title='', figsize=(40,15)):
    plt.figure(figsize=figsize)
    plt.imshow(img.astype('uint8'))
    plt.title(title)
    plt.axis('off')
    plt.show()
    plt.close()
    

root = 'ResultsOnVal'
dir_hr = os.path.join(root, 'HR')
dir_lr = os.path.join(root, 'LR')
dir_pred = os.path.join(root, 'PRED')
names = os.listdir(dir_lr)
names = np.random.permutation(names)

dh = 40
dw = 100
list_lr = []
list_hr = []
list_pred = []
for name in names[:4]:
    path_pred = os.path.join(dir_pred, name)
    img_pred = np.array(Image.open(path_pred))
    path_hr = os.path.join(dir_hr, name)
    img_hr = np.array(Image.open(path_hr))
    path_lr = os.path.join(dir_lr, name)
    img_lr = np.array(Image.open(path_lr))
    h = img_lr.shape[0]
    w = img_lr.shape[1]
    start_h = 0 if (h-dh)==0 else randint(0, h-dh-1)
    start_w = 0 if (w-dw)==0 else randint(0, w-dw-1)
    assert(start_h>=0)
    assert(start_w>=0)
    end_h = start_h + dh
    end_w = start_w + dw
    part_lr = img_lr[start_h:end_h, start_w:end_w, :]
    part_hr = img_hr[start_h*2:end_h*2, start_w*2:end_w*2, :]
    part_pred = img_pred[start_h*2:end_h*2, start_w*2:end_w*2, :]
    part_lr_large = imresize(part_lr, size=2.0, interp='bilinear')
    
    list_lr.append(part_lr_large)
    list_hr.append(part_hr)
    list_pred.append(part_pred)
    # ShowImage(part_lr_large)
    # ShowImage(part_pred)
    # ShowImage(part_hr)
    
lrs = np.stack(list_lr, axis=0)
preds = np.stack(list_pred, axis=0)
hrs = np.stack(list_hr, axis=0)
imgs = np.stack([hrs, preds, lrs], axis=1)
results = MergeImage(imgs)
ShowImage(results)


