import tensorflow as tf
from keras.layers import Layer

def SetTrainable(model, trainable):
    assert trainable in [True, False]
    for layer in model.layers:
        layer.trainable = trainable
    model.trainable = trainable
    
def FindLayerByName(model, layer_name):
    if layer_name is None: return None
    for layer, idx in zip(model.layers, range(len(model.layers))):
        if layer.name == layer_name: return idx
    return None

class PixelShuffle(Layer):
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
    












