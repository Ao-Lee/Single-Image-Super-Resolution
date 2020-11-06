import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet50V2

def Vgg(input_shape=(384,384,3)):
    vgg = VGG16(input_shape=input_shape, weights='imagenet', include_top=False)
    out_f1 = vgg.get_layer('block2_conv2').output
    out_f2 = vgg.get_layer('block3_conv3').output
    model = KM.Model(vgg.inputs, [out_f1, out_f2])
    return model

def Resnet(input_shape=(384, 384, 3)):
    resnet = ResNet50V2(input_shape=input_shape, weights='imagenet', include_top=False)
    out_f1 = resnet.get_layer('conv2_block3_1_relu').output
    out_f2 = resnet.get_layer('conv3_block4_1_relu').output
    out_f3 = resnet.get_layer('conv4_block6_1_relu').output
    model = KM.Model(resnet.inputs, [out_f1, out_f2, out_f3])
    return model

class PixelShuffle(KL.Layer):
    def __init__(self, scale, **kwargs):
        super(PixelShuffle, self).__init__(**kwargs)
        self.scale = scale

    def call(self, x):
        return tf.nn.depth_to_space(x, block_size=self.scale)
    
    def compute_output_shape(self, input_shape):
        new_height = None if input_shape[1] is None else input_shape[1]*self.scale
        new_width = None if input_shape[2] is None else input_shape[2]*self.scale
        new_channel = None if input_shape[3] is None else input_shape[3]//(self.scale**2)
        return (input_shape[0], new_height, new_width, new_channel)
    
def Downsample(filters, size, norm_type='batchnorm', apply_norm=True):
    """
    Args:
        filters: number of filters
        size: filter size
        norm_type: Normalization type; either 'batchnorm' or 'layernorm'.
        apply_norm: If True, adds the batchnorm layer

    Returns:
        Downsample Sequential Model
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(KL.Conv2D(filters, size, strides=2, padding='same',
                         kernel_initializer=initializer,
                         use_bias=False))

    if apply_norm:
        if norm_type.lower() == 'batchnorm':
            result.add(KL.BatchNormalization())
        elif norm_type.lower() == 'layernorm':
            result.add(KL.LayerNormalization())

    result.add(KL.LeakyReLU())

    return result

def Upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
    """
    Args:
        filters: number of filters
        size: filter size
        norm_type: Normalization type; either 'batchnorm' or 'layernorm'.
        apply_dropout: If True, adds the dropout layer
    """

    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(KL.Conv2DTranspose(filters, size, strides=2,
                                  padding='same',
                                  kernel_initializer=initializer,
                                  use_bias=False))

    if norm_type.lower() == 'batchnorm':
        result.add(KL.BatchNormalization())
    elif norm_type.lower() == 'layernorm':
        result.add(KL.LayerNormalization())

    if apply_dropout:
        result.add(KL.Dropout(0.5))

    result.add(KL.ReLU())

    return result

def Discriminator(input_shape=(None, None, 3), norm_type='layernorm'):
    """
    PatchGan discriminator model (https://arxiv.org/abs/1611.07004).
    norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    x = KL.Input(shape=input_shape, name='input_image')
    inputs = x

    x = Downsample(64, 4, norm_type, False)(x)  # (B, 128, 128, 64)
    x = Downsample(128, 4, norm_type)(x)        # (B, 64, 64, 128)
    x = Downsample(192, 4, norm_type)(x)        # (B, 32, 32, 256)

    output0 = KL.Conv2D(1, 4, strides=1, 
                        kernel_initializer=initializer)(x) # (B, 29, 29, 1)
    output0 = KL.GlobalAveragePooling2D()(output0) # (B, 1)
    
    
    x = KL.ZeroPadding2D()(x)    # (B, 34, 34, 256)
    x = KL.Conv2D(256, 4, strides=1, 
                     kernel_initializer=initializer,
                     use_bias=False)(x)    # (B, 31, 31, 512)

    if norm_type.lower() == 'batchnorm':
        x = KL.BatchNormalization()(x)
    elif norm_type.lower() == 'layernorm':
        x = KL.LayerNormalization()(x)

    x = KL.LeakyReLU()(x)

    x = KL.ZeroPadding2D()(x) # (B, 33, 33, 512)

    output1 = KL.Conv2D(1, 4, strides=1, 
                        kernel_initializer=initializer)(x) # (B, 30, 30, 1)
    output1 = KL.GlobalAveragePooling2D()(output1) # (B, 1)
    
    x = KL.Conv2D(256, 4, strides=1, 
                     kernel_initializer=initializer,
                     use_bias=False)(x) # (B, 30, 30, 512)
    

    if norm_type.lower() == 'batchnorm':
        x = KL.BatchNormalization()(x)
    elif norm_type.lower() == 'layernorm':
        x = KL.LayerNormalization()(x)
        
    x = KL.LeakyReLU()(x)
    x = KL.ZeroPadding2D()(x) # (B, 32, 32, 512)
    
    output2 = KL.Conv2D(1, 4, strides=1, 
                        kernel_initializer=initializer)(x)    # (B, 29, 29, 1)
    output2 = KL.GlobalAveragePooling2D()(output2) # (B, 1)
    
    outputs = output0 + output1 + output2
    return KM.Model(inputs=inputs, outputs=outputs)

'''
def Discriminator(input_shape=(None, None, 3), norm_type='layernorm'):
    """
    PatchGan discriminator model (https://arxiv.org/abs/1611.07004).
    norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    inp = KL.Input(shape=input_shape, name='input_image')
    x = inp

    down1 = Downsample(64, 4, norm_type, False)(x)    # (bs, 128, 128, 64)
    down2 = Downsample(128, 4, norm_type)(down1)    # (bs, 64, 64, 128)
    down3 = Downsample(256, 4, norm_type)(down2)    # (bs, 32, 32, 256)

    zero_pad1 = KL.ZeroPadding2D()(down3)    # (bs, 34, 34, 256)
    conv = KL.Conv2D(512, 4, strides=1, 
                     kernel_initializer=initializer,
                     use_bias=False)(zero_pad1)    # (bs, 31, 31, 512)

    if norm_type.lower() == 'batchnorm':
        norm1 = KL.BatchNormalization()(conv)
    elif norm_type.lower() == 'layernorm':
        norm1 = KL.LayerNormalization()(conv)

    leaky_relu = KL.LeakyReLU()(norm1)

    zero_pad2 = KL.ZeroPadding2D()(leaky_relu)    # (bs, 33, 33, 512)

    last = KL.Conv2D(1, 4, strides=1, 
                     kernel_initializer=initializer)(zero_pad2)    # (bs, 30, 30, 1)

    return KM.Model(inputs=inp, outputs=last)
'''

if __name__=='__main__':
    # d = Discriminator(input_shape=(256, 256, 3))
    model = Resnet(input_shape=(128, 128, 3))