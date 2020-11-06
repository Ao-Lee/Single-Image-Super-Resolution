import tensorflow as tf
from functools import partial
import tensorflow.keras.models as KM
from mymodel.loss import GradientPenalty, FeatLoss, MAE, WLossPos, WLossNeg

class WGanGP(KM.Model):
    '''
    Wasserstein GAN (gradient penalty) based trainer for single image super resolution
    there are some requirements (or recommendations) for networks used in w-gan.
        (1) Use ReLU and Tanh activations in G 
        
        (2) Use leaky ReLUs in D.
        
        (2) Do not use pooling layers in G or D network, use convolutional layers 
        with stride 2 for downsampling and transpose conv for upsampling instead.
    
        (3) both G and D networks should have normalization layers. batch norm layer is 
        recommended. But if Gradient Penalty is applied, batch norm is no longer available,
        Since layer normalization is more stable than instance normalization, it is recommand
        to use layer normalization in D network if Gradient Penalty is applied.
        
        (4) patch GAN with different(or pyramid?) receptive field can be used.
        
    some thoughts why batch norm can not be applied in gradient penalty based discriminator:
        gradient penalty term directly otimizes the gradient loss of each vector 
        sampled between data distribution and generated distribution, each of the 
        vectors has its gradient and the gradient is independent of all other vectors, 
        so the gradient penalty must be calculated separately w.r.t. each sampled 
        vector. If batch normalization is applied in discriminator, the region 
        constrained by 1-Lipchitz would be somewhere else instead of "the region 
        between data distribution and generated distribution".
    '''
    def __init__(self, discriminator, generator, featnet, opt):
        super(WGanGP, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.featnet = featnet # used to get the feature reconstruction loss
        # self.featlen = len(featnet) # number of tensors that the feature net output
        self.opt = opt
        
        assert hasattr(opt, 'weightGP')
        assert hasattr(opt, 'weightAdv')
        assert hasattr(opt, 'weightFeat')
        assert hasattr(opt, 'weightPixel')

    def compile(self, optimizer_d, optimizer_g):
        super(WGanGP, self).compile()
        self.optimizer_d = optimizer_d
        self.optimizer_g = optimizer_g

    def ComputeLoss(self, data, training=True):
        img_lr, img_hr = data
        img_fake = self.generator(img_lr, training=training)

        feat_fake = self.featnet(img_fake) # a list of tensors
        feat_hr = self.featnet(img_hr) # a list of tensors

        disc_real = self.discriminator(img_hr, training=training)
        disc_fake = self.discriminator(img_fake, training=training)

        gp = GradientPenalty(partial(self.discriminator, training=training), img_hr, img_fake) * self.opt.weightGP
        loss_d = WLossPos(disc_fake) + WLossNeg(disc_real) + gp 
    
        loss_adv = WLossNeg(disc_fake) * self.opt.weightAdv
        loss_feat0 = FeatLoss(feat_fake[0], feat_hr[0]) * self.opt.weightFeat
        loss_feat1 = FeatLoss(feat_fake[1], feat_hr[1]) * self.opt.weightFeat
        loss_feat2 = FeatLoss(feat_fake[2], feat_hr[2]) * self.opt.weightFeat
        
        loss_mae = MAE(img_fake, img_hr) * self.opt.weightPixel
        loss_g = loss_adv + loss_feat0 + loss_feat1 + loss_feat2 + loss_mae
        return loss_d, loss_g, gp, loss_adv, loss_feat0, loss_feat1, loss_feat2, loss_mae
        
    def train_step(self, data):  
        with tf.GradientTape() as tape_g, tf.GradientTape() as tape_d:
            loss_d, loss_g, gp, loss_adv, loss_feat0, loss_feat1, loss_feat2, loss_mae = self.ComputeLoss(data, training=True)
            
        gradients_g = tape_g.gradient(loss_g, self.generator.trainable_variables)
        gradients_d = tape_d.gradient(loss_d, self.discriminator.trainable_variables)

        self.optimizer_g.apply_gradients(zip(gradients_g, self.generator.trainable_variables))
        self.optimizer_d.apply_gradients(zip(gradients_d, self.discriminator.trainable_variables))
        
        result = {'loss_d': loss_d,
                  'loss_g': loss_g, 
                  'gp': gp, 
                  'loss_adv': loss_adv, 
                  'loss_feat0': loss_feat0,
                  'loss_feat1': loss_feat1, 
                  'loss_feat2': loss_feat2, 
                  'loss_mae': loss_mae}
        return result
    
    def predict_step(self, data):
        if isinstance(data, tuple) and len(data) == 1:
            data = data[0]
        img_lr = data
        img_fake = self.generator(img_lr, training=False)
        return img_fake

    def test_step(self, data):
        loss_d, loss_g, gp, loss_adv, loss_feat0, loss_feat1, loss_feat2, loss_mae = self.ComputeLoss(data, training=False)
        result = {'loss_d': loss_d,
                  'loss_g': loss_g, 
                  'gp': gp, 
                  'loss_adv': loss_adv, 
                  'loss_feat0': loss_feat0,
                  'loss_feat1': loss_feat1,
                  'loss_feat2': loss_feat2, 
                  'loss_mae': loss_mae}
        return result
    
if __name__=='__main__':
    from mymodel.srgen import SrGen
    from mymodel.utils import Vgg, Resnet, Discriminator
    import argparse
    opt = argparse.Namespace()
    opt.weightAdv = 0.1
    opt.weightGP = 10
    opt.weightFeat = 10
    opt.weightPixel = 0.002
    
    optimizer_g = tf.keras.optimizers.RMSprop(1e-4)
    optimizer_d = tf.keras.optimizers.RMSprop(1e-4)
    shape_low = (64, 64, 3)
    shape_high = (128, 128, 3)
    
    generator = SrGen(input_shape=shape_low)
    # generator.load_weights(os.path.join('save_model', 'weights.h5'))
    discriminator = Discriminator(input_shape=shape_high)
    feat = Resnet(input_shape=shape_high)
    model = WGanGP(discriminator, generator, feat, opt)
    model.compile(optimizer_g=optimizer_g, optimizer_d=optimizer_d)