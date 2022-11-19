import tensorflow as tf
import numpy as np
from network.resnet50 import ResNet50

# Normalization
def normalize(x):
    x_min = tf.reduce_min(input_tensor=tf.reduce_min(input_tensor=x, axis=1, keepdims=True), axis=2, keepdims=True)
    x_max = tf.reduce_max(input_tensor=tf.reduce_max(input_tensor=x, axis=1, keepdims=True), axis=2, keepdims=True)
    return (x - x_min) / (x_max - x_min + np.finfo('float32').eps)

# output from resnet50
def build(img_320, is_training):
    # Define the interpolation
    bilinear = tf.image.ResizeMethod.BILINEAR

    # Get the output shape
    o_shape = tf.shape(input=img_320)

    # Resize image
    size = [320, 320]
    resized_img = tf.image.resize(img_320, size, method=bilinear)

    # Feed to resnet and get output
    with tf.compat.v1.variable_scope('Network'):
        with tf.compat.v1.variable_scope('Depth'):
            resnet_50 = ResNet50({'data': resized_img-0.5}, is_training)

    # Get the current output
    predict = resnet_50.curr_node()
    
    # resize and normalize
    shape = o_shape[1:3]
    predict = normalize(tf.image.resize(predict, shape, method=bilinear))
    
    return predict, resnet_50
