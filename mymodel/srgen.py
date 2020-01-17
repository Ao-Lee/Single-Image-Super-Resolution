from keras.layers import Input, Conv2D
from keras.layers import BatchNormalization, Add, LeakyReLU
from keras.models import Model
from keras.initializers import RandomNormal

try:
    from model_utils import PixelShuffle
except ImportError:
    from .model_utils import PixelShuffle

'''
note that origin paper use PReLU. 
But PReLU has a bug in keras. It does not support free size input:
    SrGen(input_shape=(None, None, 3))
So here we use LeakyReLU instead
'''
def _GetWInit():
    return RandomNormal(stddev=0.02)

def _GetGInit():
    return RandomNormal(mean=1.0, stddev=0.02)

def SrGen(input_shape=(64,64,3), ratio=2):
    def ResBlock(inputs):
        d = Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer=_GetWInit())(inputs)
        d = LeakyReLU()(d)
        d = BatchNormalization(momentum=0.95, gamma_initializer=_GetGInit())(d)
        d = Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer=_GetWInit())(d)
        d = BatchNormalization(momentum=0.95, gamma_initializer=_GetGInit())(d)
        d = Add()([d, inputs])
        return d

    def Upsample(inputs):
        u = Conv2D(256, kernel_size=3, padding='same', kernel_initializer=_GetWInit())(inputs)
        u = PixelShuffle(scale=2)(u)
        u = LeakyReLU()(u)
        return u

    img_lr = Input(input_shape)
    c1 = Conv2D(64, kernel_size=9, strides=1, padding='same', kernel_initializer=_GetWInit())(img_lr)
    c1 = LeakyReLU()(c1)
    c2 = c1
    for _ in range(16):
        c2 = ResBlock(c2)

    c2 = Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer=_GetWInit())(c2)
    c2 = BatchNormalization(momentum=0.95, gamma_initializer=_GetGInit())(c2)
    c2 = Add()([c2, c1])
    u = Upsample(c2)
    if ratio == 4: u = Upsample(u)
    img_hr = Conv2D(3, kernel_size=9, strides=1, padding='same', activation='tanh', kernel_initializer=_GetWInit())(u)
    return Model(img_lr, img_hr)


if __name__=='__main__':
    import keras.backend as K
    K.clear_session()
    model = SrGen(input_shape=(None, None, 3))
