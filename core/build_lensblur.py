import tensorflow as tf
from network.kernel_net import KernelNet
from network.feature_net import FeaNet
import numpy as np

# Output from the lensblur module
def build(img_320, depth_320, is_training):

    # Get kernel net and feature net
    with tf.compat.v1.variable_scope('Network'):
        # Feed to kernel net to get the kernel
        with tf.compat.v1.variable_scope('Lensblur'):
            ker_net = KernelNet({'image': img_320-0.5, 'depth': depth_320}, is_training)
            kernel = ker_net.curr_node()
        # Feed to feature net to get the feature
        with tf.compat.v1.variable_scope('Feature'):
            fea_net = FeaNet({'image': img_320}, is_training)
            feature = fea_net.curr_node()

    # Output from kernel net and feature net
    # Stack feature
    feature = tf.stack([feature[:,:,:,::3], feature[:,:,:,1::3], feature[:,:,:,2::3]], axis=0)

    # Normalize kernel
    kernel = kernel / (tf.reduce_sum(kernel, axis=3, keepdims=True) + np.finfo("float").eps)

    # Depth of field with resolution 320
    dof_320 = tf.reduce_sum(kernel * feature, axis=4)
    dof_320 = tf.transpose(dof_320, perm=[1,2,3,0])

    return dof_320, ker_net, fea_net

