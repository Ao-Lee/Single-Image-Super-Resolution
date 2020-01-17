from keras.callbacks import ModelCheckpoint
import os
from os.path import join
import matplotlib.pyplot as plt

class Logger(object):
    def __init__(self, filename):
        self.filename = filename
        
    def LogAndPrint(self, msg):
        # 把训练信息打印在屏幕上的同时，也同步到txt文件里
        print(msg)
        if os.path.exists(self.filename):
            mod = 'a' # append if already exists
        else:
            mod = 'w' # make a new file if not
        with open(self.filename, mod) as file:
            file.write(msg)
            file.write('\n')
            
'''
inputs:
    list_imgs:
        a list of batch of images, for example [img1, img2, ... ,imgn]
        each img in the list has shape of (batch, h, w, c)
    list_names:
        a list of strings [img1_title, img2_title, ... ,imgn_title]
    return:
        plot a figure with following structure:
        -------------------------------------------------
        |img1_title     img2_title      imgn_title      |
        |-----------------------------------------------|
        |img1[batch0]   img2[batch0]    imgn[batch0]    |
        |-----------------------------------------------|
        |img1[batch1]   img2[batch1]    imgn[batch1]    |
        -------------------------------------------------
'''
def SaveImgPerEpoch(list_imgs, list_names, save_dir, filename):
    assert len(list_imgs)==len(list_names)
    num_imgs = len(list_imgs)
    os.makedirs(save_dir, exist_ok=True)
    fig, axs = plt.subplots(2, num_imgs, figsize=(20,20))
    for batch_id in range(2):
        for img_id in range(num_imgs):
            axs[batch_id, img_id].imshow(list_imgs[img_id][batch_id])
            axs[batch_id, img_id].set_title(list_names[img_id])
            axs[batch_id, img_id].axis('off')
            
    fig.savefig(join(save_dir, filename))
    plt.close()
        
'''
在使用 callbacks.ModelCheckpoint() 进行多gpu并行计算时，callbacks函数会报错
for more details, see:
    https://github.com/keras-team/keras/issues/8463
    https://www.jianshu.com/p/1d7977599e90
'''
class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self,
                 single_model,
                 filepath, 
                 monitor='val_loss', 
                 verbose=0,
                 save_best_only=False, 
                 save_weights_only=False,
                 mode='auto', 
                 period=1):
        self.single_model = single_model
        super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only,mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint, self).set_model(self.single_model)
        
    

    
    