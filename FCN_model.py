from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.applications.vgg16 import *
import tensorflow as tf

def FCN_8(IMG_SHAPE):
    
    inputs = Input(shape = IMG_SHAPE, name='input')
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
    # Building a pre-trained VGG-16 feature extractor (i.e., without the final FC layers)
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=s)
    f3 = vgg16.get_layer('block3_pool').output  
    f4 = vgg16.get_layer('block4_pool').output  
    f5 = vgg16.get_layer('block5_pool').output  

    # Replacing VGG dense layers by convolutions:
    f5_conv1 = Conv2D(filters=4086, kernel_size=7, padding='same',
                      activation='relu')(f5)
    f5_drop1 = Dropout(0.5)(f5_conv1)
    f5_conv2 = Conv2D(filters=4086, kernel_size=1, padding='same',
                      activation='relu')(f5_drop1)
    f5_drop2 = Dropout(0.5)(f5_conv2)
    f5_conv3 = Conv2D(filters=3, kernel_size=1, padding='same',
                      activation=None)(f5_drop2)
    
    # Using a transposed conv (w/ s=2) to upscale `f5` into a 14 x 14 map,so it can be merged with features from `f4_conv1` obtained from `f4`
    
    f5_conv3_x2 = Conv2DTranspose(filters=3, kernel_size=4, strides=2,
                            use_bias=False, padding='same', activation='relu')(f5)
    f4_conv1 = Conv2D(filters=3, kernel_size=1, padding='same',
                  activation=None)(f4)

    # Merging the 2 feature maps (addition):
    merge1 = add([f4_conv1, f5_conv3_x2])

    # We repeat the operation to merge `merge1` and `f3` into a 28 x 28 map:
    merge1_x2 = Conv2DTranspose(filters=3, kernel_size=4, strides=2,
                            use_bias=False, padding='same', activation='relu')(merge1)
    f3_conv1 = Conv2D(filters=3, kernel_size=1, padding='same',
                  activation=None)(f3)
    merge2 = add([f3_conv1, merge1_x2])

    # Finally, we use another transposed conv to decode and up-scale the feature map
    # to the original shape, i.e., using a stride 8 to go from 28 x 28 to 224 x 224 here:
    outputs = Conv2DTranspose(filters = 1, kernel_size=16, strides=8,
                          padding='same', activation=None)(merge2)

    fcn_model = Model(inputs, outputs)
    return fcn_model

import FCN_model 