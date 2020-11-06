import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
from tensorflow.keras.initializers import RandomNormal
from mymodel.utils import PixelShuffle

'''
note that origin paper use PReLU. 
But PReLU has a bug in keras. It does not support free size input:
    SrGen(input_shape=(None, None, 3))
So here we use LeakyReLU instead
'''

def SrGen(input_shape=(64,64,3), ratio=2):
    
    def _GetWInit():
        return RandomNormal(stddev=0.02)
    
    def _GetGInit():
        return RandomNormal(mean=1.0, stddev=0.02)

    def ResBlock(inputs):
        d = KL.Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer=_GetWInit())(inputs)
        d = KL.LeakyReLU()(d)
        d = KL.BatchNormalization()(d)
        d = KL.Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer=_GetWInit())(d)
        d = KL.BatchNormalization()(d)
        d = KL.Add()([d, inputs])
        return d

    def Upsample(inputs):
        u = KL.Conv2D(256, kernel_size=3, padding='same', kernel_initializer=_GetWInit())(inputs)
        u = PixelShuffle(scale=2)(u)
        u = KL.LeakyReLU()(u)
        return u

    img_lr = KL.Input(input_shape)
    c1 = KL.Conv2D(64, kernel_size=9, strides=1, padding='same', kernel_initializer=_GetWInit())(img_lr)
    c1 = KL.LeakyReLU()(c1)
    c2 = c1
    for _ in range(16):
        c2 = ResBlock(c2)

    c2 = KL.Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer=_GetWInit())(c2)
    c2 = KL.BatchNormalization()(c2)
    c2 = KL.Add()([c2, c1])
    u = Upsample(c2)
    if ratio == 4: u = Upsample(u)
    img_hr = KL.Conv2D(3, kernel_size=9, strides=1, padding='same', activation='tanh', kernel_initializer=_GetWInit())(u)
    return KM.Model(img_lr, img_hr)


if __name__=='__main__':
    # model = SrGen(input_shape=(None, None, 3))
    model = SrGen(input_shape=(128, 128, 3), ratio=2)
    model.summary()
