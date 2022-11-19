from network.network import Network
import tensorflow as tf

# SRNet
class SRNet(Network):
    # Initialization
    def __init__(self, input, is_training, trainable=True, bn_global=None, upsample_size=640):
        self.is_training = is_training
        self.bn_global = bn_global
        self.upsample_size = upsample_size
        super(SRNet, self).__init__(input, trainable)

    def setup(self):
        # lr layers
        (self.feed('lr_in')
        .conv(3, 3, 32, 1, 1, 1, name='lr1_1', relu=False, biased=False)
        .batch_normalization(name='lr1_1_bn', relu=True, is_training=self.is_training)
        .conv(3, 3, 64, 2, 2, 1, name='lr2_1', relu=False, biased=False)
        .batch_normalization(name='lr2_1_bn', relu=True, is_training=self.is_training)
        .conv(3, 3, 128, 2, 2, 1, name='lr3_1', relu=False, biased=False)
        .batch_normalization(name='lr3_1_bn', relu=True, is_training=self.is_training)
        .conv(3, 3, 256, 2, 2, 1, name='lr4_1', relu=False, biased=False)
        .batch_normalization(name='lr4_1_bn', relu=True, is_training=self.is_training)
        .conv(3, 3, 256, 1, 1, 1, name='lr4_2', relu=False, biased=False)
        .batch_normalization(name='lr4_2_bn', relu=True, is_training=self.is_training)
        .conv(3, 3, 128, 1, 1, 1, name='lr5_1', relu=False, biased=False)
        .batch_normalization(name='lr5_1_bn', relu=True, is_training=self.is_training)
        .resize(shape=tf.shape(input=self.layers['lr3_1_bn'])[1:3], name='lr5_1_up')
        # Addition of some nodes
        .feed('lr3_1_bn', 'lr5_1_up')
        .add(name='lr5_1_add')
        .conv(3, 3, 64, 1, 1, 1, name='lr6_1', relu=False, biased=False)
        .batch_normalization(name='lr6_1_bn', relu=True, is_training=self.is_training)
        .resize(shape=tf.shape(input=self.layers['lr2_1'])[1:3], name='lr6_1_up')
        .feed('lr2_1_bn', 'lr6_1_up')
        .add(name='lr6_1_add')
        .conv(3, 3, 32, 1, 1, 1, name='lr7_1', relu=False, biased=False)
        .batch_normalization(name='lr7_1_bn', relu=True, is_training=self.is_training)
        .resize(shape=tf.shape(input=self.layers['lr1_1'])[1:3], name='lr7_1_up')
        .feed('lr1_1_bn', 'lr7_1_up')
        .add(name='lr7_1_add')
        .conv(3, 3, 32, 1, 1, 1, name='lr8_1', relu=False, biased=False)
        .batch_normalization(name='lr8_1_bn', relu=True, is_training=self.is_training)
        .resize(shape=tf.shape(input=self.layers['hr_in'])[1:3], name='lr8_1_up')
        .conv(3, 3, 32, 1, 1, 1, name='lr9_1', relu=False, biased=False)
        .batch_normalization(name='lr9_1_bn', relu=True, is_training=self.is_training)
        )

        # Predict kernel
        (self.feed('hr_in', 'lr9_1_bn')
        .concat(name='pre0')
        .conv(3, 3, 32, 1, 1, 1, name='pre1_1', relu=False, biased=False)
        .batch_normalization(name='pre1_1_bn', relu=True, is_training=self.is_training)
        .conv(3, 3, 32, 1, 1, 1, name='pre1_2', relu=False, biased=False)
        .batch_normalization(name='pre1_2_bn', relu=True, is_training=self.is_training)
        .conv(3, 3, 32, 1, 1, 1, name='pre1_3', relu=False, biased=False)
        .batch_normalization(name='pre1_3_bn', relu=True, is_training=self.is_training)
        .conv(3, 3, 32, 1, 1, 1, name='pre1_4', relu=False, biased=False)
        .batch_normalization(name='pre1_4_bn', relu=True, is_training=self.is_training)
        .conv(3, 3, 32, 1, 1, 1, name='pre1_5', relu=True, biased=True)
        )

        # Output image feature
        (self.feed('dof_640', 'hr_in')
        .concat(name='fea0')
        .padding(padding=1, name='fea0-pad', mode='SYMMETRIC')
        .conv(3, 3, 30, 1, 1, 1, name='fea1', padding='VALID', relu=True, biased=True)
        .padding(padding=2, name='fea1-pad', mode='SYMMETRIC')
        .conv(5, 5, 30, 1, 1, 1, name='fea2', padding='VALID', relu=True, biased=True)
        .padding(padding=2, name='fea2-pad', mode='SYMMETRIC')
        .conv(5, 5, 15, 1, 1, 1, name='fea3', padding='VALID', relu=True, biased=True)
        .padding(padding=2, name='fea3-pad', mode='SYMMETRIC')
        .conv(5, 5, 15, 1, 1, 1, name='fea4', padding='VALID', relu=True, biased=True)
        .feed('fea0', 'fea1', 'fea2', 'fea3', 'fea4')
        .concat(name='fea_output')
        )
