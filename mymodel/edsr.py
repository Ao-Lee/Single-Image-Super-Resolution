import numpy as np

from keras.layers import Conv2D, Input, Lambda, Add
from keras.models import Model

try:
    from model_utils import PixelShuffle
except ImportError:
    from .model_utils import PixelShuffle

def EDSR(scale=4, input_shape=(None, None, 3), num_filters=64, num_res_blocks=8, scaling=None):
    inputs = Input(shape=input_shape)
    # x = Lambda(normalize)(inputs)

    x = Conv2D(num_filters, kernel_size=3, padding='same')(inputs)
    res = x
    for i in range(num_res_blocks):
        x = ResBlock(x, num_filters, scaling)
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = Add()([x, res])

    x = Upsample(x, scale, num_filters=32)
    x = Conv2D(3, 3, padding='same', activation='tanh')(x)

    # x = Lambda(denormalize)(x)
    return Model(inputs, x, name="edsr")

def ResBlock(inputs, filters, scaling=None):
    x = Conv2D(filters, kernel_size=3, padding='same', activation='relu')(inputs)
    x = Conv2D(filters, kernel_size=3, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([inputs, x])
    return x

def Upsample(x, scale, num_filters):
    if scale == 2:
        x = Conv2D(num_filters*4, kernel_size=3, padding='same')(x)
        x = PixelShuffle(scale=2)(x)
    if scale == 3:
        x = Conv2D(num_filters*9, kernel_size=3, padding='same')(x)
        x = PixelShuffle(scale=2)(x)
    if scale == 4:
        x = Conv2D(num_filters*4, kernel_size=3, padding='same')(x)
        x = PixelShuffle(scale=2)(x)
        x = Conv2D(num_filters*4, kernel_size=3, padding='same')(x)
        x = PixelShuffle(scale=2)(x)
    return x
    
if __name__=='__main__':
    import keras.backend as K
    K.clear_session()
    model = EDSR(scale=4, input_shape=(96, 96, 3), num_filters=96, num_res_blocks=16, scaling=None)
    model.summary()