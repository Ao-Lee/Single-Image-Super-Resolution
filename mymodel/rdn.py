from keras.models import Model
from keras.layers import Input
from keras.layers import Add, Conv2D
from keras.layers import Concatenate, Lambda
from keras.regularizers import l2

try:
    from model_utils import PixelShuffle
except ImportError:
    from .model_utils import PixelShuffle

def RDB_blocks(x, name='', count=6, g=32, scaling=None, **kwargs):
    kernel_initializer = kwargs.setdefault('kernel_initializer', 'he_normal')
    kernel_regularizer = kwargs.setdefault('reg', 1e-4)
    scaling = kwargs.setdefault('scaling', None)
    ## 6 layers of RDB block
    ## this thing need to be in a damn loop for more customisability
    li = [x]
    pas = Conv2D(filters=g, 
                 kernel_size=(3,3), 
                 strides=(1,1), 
                 padding='same', 
                 activation='relu', 
                 name=name+'_conv1', 
                 kernel_initializer=kernel_initializer,
                 kernel_regularizer=l2(kernel_regularizer))(x)
    
    for i in range(2 , count+1):
        li.append(pas)
        out = Concatenate(axis=-1)(li) # conctenated out put
        pas = Conv2D(filters=g, 
                     kernel_size=(3,3), 
                     strides=(1,1), 
                     padding='same', 
                     activation='relu', 
                     name=name+'_conv'+str(i), 
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=l2(kernel_regularizer))(out)
    
    # feature extractor from the dense net
    li.append(pas)
    out = Concatenate(axis=-1)(li)
    feat = Conv2D(filters=64, 
                  kernel_size=(1,1), 
                  strides=(1, 1), 
                  padding='same',
                  activation='relu', 
                  name=name+'_Local_Conv',
                  kernel_initializer=kernel_initializer,
                  kernel_regularizer=l2(kernel_regularizer))(out)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    feat = Add()([feat , x])
    return feat
        
def Upsample(x, scale, num_filters, **kwargs):
    kernel_initializer = kwargs.setdefault('kernel_initializer', 'he_normal')
    kernel_regularizer = kwargs.setdefault('reg', 1e-4)
    if scale == 2:
        x = Conv2D(num_filters*4, kernel_size=3, padding='same', kernel_initializer=kernel_initializer, kernel_regularizer=l2(kernel_regularizer))(x)
        x = PixelShuffle(scale=2)(x)
    if scale == 3:
        x = Conv2D(num_filters*9, kernel_size=3, padding='same', kernel_initializer=kernel_initializer, kernel_regularizer=l2(kernel_regularizer))(x)
        x = PixelShuffle(scale=2)(x)
    if scale == 4:
        x = Conv2D(num_filters*4, kernel_size=3, padding='same', kernel_initializer=kernel_initializer, kernel_regularizer=l2(kernel_regularizer))(x)
        x = PixelShuffle(scale=2)(x)
        x = Conv2D(num_filters*4, kernel_size=3, padding='same', kernel_initializer=kernel_initializer, kernel_regularizer=l2(kernel_regularizer))(x)
        x = PixelShuffle(scale=2)(x)
    return x

def RDN(input_shape=(96, 96, 3), RDB_count=20 , scale=4, scaling=None, **kwargs):
    kernel_initializer = kwargs.setdefault('kernel_initializer', 'he_normal')
    kernel_regularizer = kwargs.setdefault('reg', 1e-4)
    
    inputs = Input(shape=input_shape)
    pass1 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same' , activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=l2(kernel_regularizer))(inputs)
    pass2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same' , activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=l2(kernel_regularizer))(pass1)
    
    x = pass2
    blocks_list = []
    for i in range(RDB_count):
        x = RDB_blocks(x, name='block'+str(i), scaling=scaling, **kwargs)
        blocks_list.append(x)
        
    out = Concatenate(axis = -1)(blocks_list)
    out = Conv2D(filters=64, kernel_size=(1,1), strides=(1,1), padding='same', kernel_initializer=kernel_initializer, kernel_regularizer=l2(kernel_regularizer))(out)
    out = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer=kernel_initializer, kernel_regularizer=l2(kernel_regularizer))(out)

    output = Add()([out , pass1])
    
    output = Upsample(output, scale, num_filters=32, **kwargs)
    output = Conv2D(filters=3, kernel_size=(3,3), activation='tanh', padding='same', kernel_initializer=kernel_initializer)(output)

    model = Model(inputs=inputs, outputs = output)
    return model


if __name__=='__main__':
    import keras.backend as K
    K.clear_session()
    model = RDN(input_shape=(96, 96, 3), scale=4, scaling=0.8, reg=1e-4)
    print(model.summary())
    
    





