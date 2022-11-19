from network.network import Network
import tensorflow as tf

# kernel estimation net
class KernelNet(Network):
    def __init__(self, input, is_training, trainable=True):
        # Initialization
        self.is_training = is_training
        super(KernelNet, self).__init__(input, trainable)

    def setup(self):
        # Image depth map
        (self.feed('image', 'depth')
        .concat(3, name='im_depth')
        .conv(7, 7, 64, 1, 1, 1,name='conv1_shallow', relu=False, biased=False)
        .batch_normalization(name='bn1_shallow', relu=True, is_training=self.is_training))

        # Depth
        (self.feed('depth')
         .conv(7, 7, 64, 4, 4, 1, name='conv1', relu=False, biased=False)
         .batch_normalization(name='bn1', relu=True, is_training=self.is_training))

        # Res2a 80 128
        (self.conv(1, 1, 128, 1, 1, 1, name='res2a_branch1', relu=False, biased=False)
         .batch_normalization(name='bn2a_branch1', relu=True, is_training=self.is_training))
        (self.feed('bn1')
         .conv(1, 1, 64, 1, 1, 1, name='res2a_branch2a', relu=False, biased=False)
         .batch_normalization(name='bn2a_branch2a', relu=True, is_training=self.is_training)
         .conv(3, 3, 64, 1, 1, 1, name='res2a_branch2b', relu=False, biased=False)
         .batch_normalization(name='bn2a_branch2b', relu=True, is_training=self.is_training)
         .conv(1, 1, 128, 1, 1, 1, name='res2a_branch2c', relu=False, biased=False)
         .batch_normalization(name='bn2a_branch2c', relu=False, is_training=self.is_training)
         .feed('bn2a_branch1', 'bn2a_branch2c')
         .add(name='res2a')
         .relu(name='res2a_relu'))

        # Res2b 40 256
        (self.conv(1, 1, 256, 2, 2, 1, name='res2b_branch1', relu=False, biased=False)
         .batch_normalization(name='bn2b_branch1', relu=True, is_training=self.is_training))
        (self.feed('res2a_relu')
         .conv(1, 1, 64, 1, 1, 1, name='res2b_branch2a', relu=False, biased=False)
         .batch_normalization(name='bn2b_branch2a', relu=True, is_training=self.is_training)
         .conv(3, 3, 64, 1, 1, 1, relu=False, biased=False, name='res2b_branch2b')
         .batch_normalization(name='bn2b_branch2b', relu=True, is_training=self.is_training)
         .conv(1, 1, 256, 2, 2, 1, name='res2b_branch2c', relu=False, biased=False)
         .batch_normalization(name='bn2b_branch2c', relu=False, is_training=self.is_training)
         .feed('bn2b_branch1', 'bn2b_branch2c')
         .add(name='res2b')
         .relu(name='res2b_relu'))

        # Res3a 40 256
        (self.conv(1, 1, 64, 1, 1, 1, name='res3a_branch2a', relu=False, biased=False)
         .batch_normalization(name='bn3a_branch2a', relu=True, is_training=self.is_training)
         .conv(3, 3, 64, 1, 1, 1, name='res3a_branch2b', relu=False, biased=False, )
         .batch_normalization(name='bn3a_branch2b', relu=True, is_training=self.is_training)
         .conv(1, 1, 256, 1, 1, 1, name='res3a_branch2c', relu=False, biased=False)
         .batch_normalization(name='bn3a_branch2c', relu=False, is_training=self.is_training)
         .feed('res2b_relu', 'bn3a_branch2c')
         .add(name='res3a')
         .relu(name='res3a_relu'))

        # Upsampling
        self.resize(shape=tf.shape(input=self.layers['image'])[1:3], name='upsample')

        # Concate upsampled and shallow
        (self.feed('upsample', 'bn1_shallow')
         .concat(3, name='concat'))

        # Res4a
        (self.conv(1, 1, 128, 1, 1, 1, name='res4a_branch1', relu=False, biased=False)
            .batch_normalization(name='bn4a_branch1', relu=True, is_training=self.is_training))
        (self.feed('concat')
         .conv(1, 1, 64, 1, 1, 1, name='res4a_branch2a', relu=False, biased=False, )
         .batch_normalization(name='bn4a_branch2a', relu=True, is_training=self.is_training)
         .conv(3, 3, 64, 1, 1, 1, name='res4a_branch2b', relu=False, biased=False)
         .batch_normalization(name='bn4a_branch2b', relu=True, is_training=self.is_training)
         .conv(1, 1, 128, 1, 1, 1, name='res4a_branch2c', relu=False, biased=False)
         .batch_normalization(name='bn4a_branch2c', relu=False, is_training=self.is_training)
         .feed('bn4a_branch1', 'bn4a_branch2c')
         .add(name='res4a')
         .relu(name='res4a_relu'))

        # Res4b
        (self.conv(1, 1, 256, 1, 1, 1, name='res4b_branch1', relu=False, biased=False)
         .batch_normalization(name='bn4b_branch1', relu=True, is_training=self.is_training))
        (self.feed('res4a_relu')
         .conv(1, 1, 64, 1, 1, 1, name='res4b_branch2a', relu=False, biased=False)
         .batch_normalization(name='bn4b_branch2a', relu=True, is_training=self.is_training)
         .conv(3, 3, 64, 1, 1, 1, name='res4b_branch2b', relu=False, biased=False)
         .batch_normalization(name='bn4b_branch2b', relu=True, is_training=self.is_training)
         .conv(1, 1, 256, 1, 1, 1, name='res4b_branch2c', relu=False, biased=False, )
         .batch_normalization(name='bn4b_branch2c', relu=False, is_training=self.is_training)
         .feed('bn4b_branch1', 'bn4b_branch2c')
         .add(name='res4b')
         .relu(name='res4b_relu'))

        (self.feed('res4b_relu')
         .conv(1, 1, 64, 1, 1, 1, name='res4c_branch2a', relu=False, biased=False)
         .batch_normalization(name='bn4c_branch2a', relu=True, is_training=self.is_training)
         .conv(3, 3, 64, 1, 1, 1, name='res4c_branch2b', relu=False, biased=False)
         .batch_normalization(name='bn4c_branch2b', relu=True, is_training=self.is_training)
         .conv(1, 1, 256, 1, 1, 1, name='res4c_branch2c', relu=False, biased=False)
         .batch_normalization(name='bn4c_branch2c', relu=False, is_training=self.is_training)
         .feed('res4b_relu', 'bn4c_branch2c')
         .add(name='res4c')
         .relu(name='res4c_relu'))
        (self.conv(1, 1, 31, 1, 1, 1, name='output', relu=False, biased=True)
         .relu(name='output_relu'))