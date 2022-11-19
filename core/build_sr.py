import tensorflow as tf
from network.srnet import SRNet
import numpy as np

# Output from srnet
def build(img_320, depth_320, dof_320, img_640, training):
    # concate 3 channels
    lr_in = tf.concat([img_320, dof_320, depth_320], axis=3)
    hr_in = img_640
    shape = tf.shape(hr_in)

    # Get DoF
    dof_640_bc = tf.image.resize(tf.cast(dof_320*255, tf.uint8), size=[shape[1], shape[2]], method=tf.image.ResizeMethod.BILINEAR)
    dof_640_bc = tf.cast(dof_640_bc, dtype=tf.float32) / 255.0

    # Get sr net
    with tf.compat.v1.variable_scope('Network', reuse=tf.compat.v1.AUTO_REUSE):
        with tf.compat.v1.variable_scope('SR'):
            sr_net = SRNet({'lr_in': lr_in, 'hr_in': hr_in, 'dof_640': dof_640_bc}, training, bn_global=True)
        # Get the corresponding feature and weigth from layer
        sr_feature = sr_net.layers['fea_output']
        sr_weight = sr_net.layers['pre1_5']

        # Output from srnet
        # Stack feature
        sr_feature = tf.stack([sr_feature[:,:,:,::3], sr_feature[:,:,:,1::3], sr_feature[:,:,:,2::3]], axis=0)
        # Normalize weights
        sr_weight = sr_weight / (tf.reduce_sum(sr_weight, axis=3, keepdims=True) + np.finfo("float").eps)
        dof_640 = tf.reduce_sum(sr_weight * sr_feature, axis=4)
        dof_640 = tf.transpose(dof_640, perm=[1,2,3,0])
        
        return dof_640, sr_net

