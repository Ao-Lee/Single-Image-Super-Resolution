from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.layers import BatchNormalization, LeakyReLU, Conv2D, Input

from keras.applications import VGG19
from keras.models import Model
from keras.optimizers import RMSprop

import numpy as np
import os
from os.path import join
import keras.backend as K
from mydata.loader_coco import COCO
from mydata.utils import InvertedProcess
from mymodel.srgen import SrGen
from mymodel.model_utils import FindLayerByName, SetTrainable
from train_utils import SaveImgPerEpoch, Logger
import cfg

def GetVggModel(input_shape=(384,384,3)):
    vgg = VGG19(input_shape=input_shape, weights='imagenet', include_top=False)
    
    idx_f1 = FindLayerByName(vgg, 'block5_conv2')
    out_f1 = vgg.layers[idx_f1].output
    idx_f2 = FindLayerByName(vgg, 'block5_conv4')
    out_f2 = vgg.layers[idx_f2].output  
    
    model = Model(vgg.inputs, [out_f1, out_f2])
    SetTrainable(model, False)
    return model    

def GetG(input_shape=(96,96,3), ratio=2):
    model = SrGen(input_shape=input_shape, ratio=ratio)
    return model

def GetD(input_shape=(384,384,3)):
    def d_block(layer_input, filters, strides=1, bn=True):
        d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    inputs = Input(shape=input_shape)
    x = d_block(inputs, 64, strides=2, bn=False)
    x = d_block(x, 64)
    x = d_block(x, 64, strides=2)
    x = d_block(x, 64*2)
    x = d_block(x, 64*2, strides=2)
    x = d_block(x, 64*4)
    x = d_block(x, 64*4, strides=2)
    x = d_block(x, 64*8)
    x = d_block(x, 64*8, strides=2)
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    # x = Flatten()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, name='outputs')(x)
    return Model(inputs, outputs)
    
class WGAN_Trainer():
    def __init__(self, loss_weights=[1, 0.1, 0.1, 0], lr=0.0001, batch=5, model_path=None):
        '''
        loss_weights are weights of the following losses:
            [1] WassersteinLoss
            [2] VGG feature mse loss 
            [3] VGG feature mse loss 
            [4] L1 loss between fake hr and real hr images (naïve method)
        '''
        self.batch = batch
        optimizer = RMSprop(lr=lr)
        
        # data
        ds = COCO(root=cfg.PATH_COCO, batch_size=batch, shape_lr=cfg.SIZE_LR, ratio=2)
        self.shape_lr = ds.GetShapeLowResolution()
        self.shape_hr = ds.GetShapeHighResolution()
        self.data_loader = ds.GetGenerator_Tr()

        # model-vgg
        self.vgg = GetVggModel(input_shape=self.shape_hr)
        self.vgg.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        # model-D
        self.discriminator = GetD(input_shape=self.shape_hr)
        self.discriminator.compile(loss=self.WassersteinLoss, optimizer=optimizer, metrics=['accuracy'])
        
        # model-G
        self.generator = GetG(input_shape=self.shape_lr, ratio=2)
        if model_path: self.generator.load_weights(os.path.join(cfg.PATH_MODEL, model_path))

        # model_combine
        img_lr = Input(shape=self.shape_lr)
        fake_hr = self.generator(img_lr)
        vgg_features = self.vgg(fake_hr)
        vgg_feature_0 = vgg_features[0]
        vgg_feature_1 = vgg_features[1]

        SetTrainable(self.discriminator, False)
        distance_w = self.discriminator(fake_hr)

        self.combined = Model(img_lr, [distance_w, vgg_feature_0, vgg_feature_1, fake_hr])
        self.combined.compile(loss=[self.WassersteinLoss, 'mean_squared_error', 'mean_squared_error', 'mean_absolute_error'], loss_weights=loss_weights, optimizer=optimizer)

        # others
        self.label_neg = -np.ones((self.batch, 1))
        self.label_pos = np.ones((self.batch, 1))
        self.logger = Logger('train_log.txt')
        
    def WassersteinLoss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)
    
    def TrainD(self, clip_value):
        imgs_lr, imgs_hr = next(self.data_loader)
        fake_hr = self.generator.predict(imgs_lr)
        result_real = self.discriminator.train_on_batch(imgs_hr, self.label_neg)
        d_loss_real = result_real[0]
        result_fake = self.discriminator.train_on_batch(fake_hr, self.label_pos)
        d_loss_fake = result_fake[0]
            
        # clip weights
        for l in self.discriminator.layers:
            weights = l.get_weights()
            weights = [np.clip(w, -clip_value, clip_value) for w in weights]
            l.set_weights(weights)
                
        d_loss = d_loss_fake + d_loss_real
        info_d = 'dloss:{:.5f}\tdreal:{:.5f}\tdfake:{:.5f}'.format(d_loss, d_loss_real, d_loss_fake)
        return info_d
            
    def TrainG(self):
        imgs_lr, imgs_hr = next(self.data_loader)
        features = self.vgg.predict(imgs_hr)
        result = self.combined.train_on_batch(imgs_lr, [self.label_neg, features[0], features[1], imgs_hr])
        # result = [round(num, 3) for num in list(result)]
        info_g = 'total:{:.2f}\tWasserstein:{:.5f}\tvgg:{:.2f}\tvgg:{:.2f}\tpixel:{:.3f}'.format(result[0], result[1], result[2], result[3], result[4])
        return info_g
            
    def Train(self, 
              epoches=30000,  # total number of epoches
              report_interval=50,  # save models & pics every report_interval epoches
              save_dir='results',   # directory to save models & pics
              clip_value = 0.01,    # clip value applied on gradient of the D net
              ):
              
        for epoch in range(epoches):
            n_critic = 100 if epoch % 200 == 0 else 1
            for epoch_critic in range(n_critic):
                info_d = self.TrainD(clip_value=clip_value)
                info_d = 'Epoch:{}'.format(epoch)+'\t'+info_d
                if epoch_critic % 10 == 0 and epoch_critic != 0: self.logger.LogAndPrint(info_d)
                
            info_g = self.TrainG()
            info_g = 'Epoch:{}'.format(epoch)+'\t'+info_g
            
            if epoch % report_interval != 0: continue
            self.logger.LogAndPrint(info_d)
            self.logger.LogAndPrint(info_g)

            imgs_lr, imgs_hr = next(self.data_loader)
            fake_hr = self.generator.predict(imgs_lr)
            list_imgs = [fake_hr, imgs_hr, imgs_lr]
            
            list_imgs = [InvertedProcess(img) for img in list_imgs]
            list_names = ['Generated', 'HR Img', 'LR Img']
            epoch_current_str = format(epoch, '05d')
            filename = 'wgan_pic_epoch{}.png'.format(epoch_current_str)
            SaveImgPerEpoch(list_imgs, list_names, save_dir, filename)
            filename = 'wgan_model_epoch{}.hdf5'.format(epoch_current_str)
            self.generator.save_weights(join(save_dir, filename))
                
if __name__ == '__main__':
    K.clear_session()
    
    model_path='sisr_96_srgan_pretrained_mse.hdf5'
    model_path = None
    trainer = WGAN_Trainer(lr=0.0002, loss_weights=[1, 0.1, 0.1, 0], batch=4, model_path=model_path)
    train_params = {}
    train_params['epoches'] = 30000
    train_params['report_interval'] = 500   
    train_params['save_dir'] = 'results'
    train_params['clip_value'] = 0.01
    # trainer.Train(**train_params)
    

    
    
    