import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
import warnings
warnings.filterwarnings("ignore")


# from https://github.com/keras-team/keras/issues/4161
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.allow_soft_placement = True  # 如果指定的设备不存在，允许TF自动分配设备的设备不存在，允许TF自动分配设备           
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


SIZE_LR = 64 # training input size for low resolution image
NUM_GPU = len(os.environ["CUDA_VISIBLE_DEVICES"].split(sep=','))
LINUX_ENV = False
PATH_MODEL = '/home/liao/DM/Models' if LINUX_ENV else 'C:\AAAA\DM\Models'
PATH_DIV2K = '//home//liao//DM//DIV2K' if LINUX_ENV else 'C:\AAAA\DM\DIV2K'
PATH_COCO = '//home//liao//DM//COCO' if LINUX_ENV else 'C:\AAAA\DM\COCO'


