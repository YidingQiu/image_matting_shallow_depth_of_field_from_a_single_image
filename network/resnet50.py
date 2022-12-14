from network.network import Network
import tensorflow as tf

# Implementing the first 14 layer of ResNet50
class ResNet50(Network):
    def __init__(self, input, is_training, trainable=True):
         # initialization
         self.is_training = is_training
         super(ResNet50, self).__init__(input, trainable)
    
    def setup(self):
          # define method
          bilinear = tf.image.ResizeMethod.BILINEAR
          (self.feed('data')
             .padding(padding=3, name='data_pad')
             .conv(7, 7, 64, 2, 2, 1, name='conv1', relu=False, padding='VALID')
             .batch_normalization(name='bn_conv1', relu=True)
             .max_pool(3, 3, 2, 2, name='pool1')
             .conv(1, 1, 256, 1, 1, 1, name='res2a_branch1', relu=False, biased=False)
             .batch_normalization(name='bn2a_branch1'))

          (self.feed('pool1')
             .conv(1, 1, 64, 1, 1, 1, name='res2a_branch2a', relu=False, biased=False)
             .batch_normalization(name='bn2a_branch2a', relu=True)
             .conv(3, 3, 64, 1, 1, 1, name='res2a_branch2b', relu=False, biased=False)
             .batch_normalization(name='bn2a_branch2b', relu=True)
             .conv(1, 1, 256, 1, 1, 1, biased=False, relu=False, name='res2a_branch2c')
             .batch_normalization(name='bn2a_branch2c'))

          (self.feed('bn2a_branch1', 'bn2a_branch2c')
             .add(name='res2a')
             .relu(name='res2a_relu')
             .conv(1, 1, 64, 1, 1, 1, name='res2b_branch2a', relu=False, biased=False)
             .batch_normalization(name='bn2b_branch2a', relu=True)
             .conv(3, 3, 64, 1, 1, 1, name='res2b_branch2b', relu=False, biased=False)
             .batch_normalization(name='bn2b_branch2b', relu=True)
             .conv(1, 1, 256, 1, 1, 1, name='res2b_branch2c', relu=False, biased=False)
             .batch_normalization(name='bn2b_branch2c'))

          (self.feed('res2a_relu', 'bn2b_branch2c')
             .add(name='res2b')
             .relu(name='res2b_relu')
             .conv(1, 1, 64, 1, 1, 1, name='res2c_branch2a', relu=False, biased=False)
             .batch_normalization(name='bn2c_branch2a', relu=True)
             .conv(3, 3, 64, 1, 1, 1, name='res2c_branch2b', relu=False, biased=False)
             .batch_normalization(name='bn2c_branch2b', relu=True)
             .conv(1, 1, 256, 1, 1, 1, name='res2c_branch2c', relu=False,biased=False)
             .batch_normalization(name='bn2c_branch2c'))

          (self.feed('res2b_relu', 'bn2c_branch2c')
             .add(name='res2c')
             .relu(name='res2c_relu')
             .conv(1, 1, 512, 2, 2, 1, name='res3a_branch1', relu=False, biased=False)
             .batch_normalization(name='bn3a_branch1'))
         
          (self.feed('res2c_relu')
             .conv(1, 1, 128, 2, 2, 1, name='res3a_branch2a', relu=False, biased=False)
             .batch_normalization(name='bn3a_branch2a', relu=True)
             .conv(3, 3, 128, 1, 1, 1, name='res3a_branch2b', relu=False, biased=False)
             .batch_normalization(name='bn3a_branch2b', relu=True, )
             .conv(1, 1, 512, 1, 1, 1, name='res3a_branch2c', relu=False,biased=False)
             .batch_normalization(name='bn3a_branch2c'))

          (self.feed('bn3a_branch1', 'bn3a_branch2c')
             .add(name='res3a')
             .relu(name='res3a_relu')
             .conv(1, 1, 128, 1, 1, 1, name='res3b_branch2a', relu=False, biased=False)
             .batch_normalization(relu=True, name='bn3b_branch2a')
             .conv(3, 3, 128, 1, 1, 1, name='res3b_branch2b', relu=False, biased=False)
             .batch_normalization(relu=True, name='bn3b_branch2b')
             .conv(1, 1, 512, 1, 1, 1, name='res3b_branch2c', relu=False, biased=False)
             .batch_normalization(name='bn3b_branch2c'))

          (self.feed('res3a_relu', 'bn3b_branch2c')
             .add(name='res3b')
             .relu(name='res3b_relu')
             .conv(1, 1, 128, 1, 1, 1, name='res3c_branch2a', relu=False, biased=False)
             .batch_normalization(name='bn3c_branch2a', relu=True)
             .conv(3, 3, 128, 1, 1, 1, name='res3c_branch2b', relu=False, biased=False)
             .batch_normalization(name='bn3c_branch2b', relu=True)
             .conv(1, 1, 512, 1, 1, 1, name='res3c_branch2c', relu=False, biased=False)
             .batch_normalization(name='bn3c_branch2c'))

          (self.feed('res3b_relu', 'bn3c_branch2c')
             .add(name='res3c')
             .relu(name='res3c_relu')
             .conv(1, 1, 128, 1, 1, 1, name='res3d_branch2a', relu=False, biased=False)
             .batch_normalization(name='bn3d_branch2a', relu=True)
             .conv(3, 3, 128, 1, 1, 1, name='res3d_branch2b', relu=False, biased=False)
             .batch_normalization(name='bn3d_branch2b', relu=True)
             .conv(1, 1, 512, 1, 1, 1, name='res3d_branch2c', relu=False, biased=False)
             .batch_normalization(name='bn3d_branch2c'))

          (self.feed('res3c_relu', 'bn3d_branch2c')
             .add(name='res3d')
             .relu(name='res3d_relu')
             .conv(1, 1, 1024, 1, 1, 1, name='res4a_branch1', relu=False, biased=False)
             .batch_normalization(name='bn4a_branch1'))

          (self.feed('res3d_relu')
             .conv(1, 1, 256, 1, 1, 1, name='res4a_branch2a', relu=False, biased=False)
             .batch_normalization(name='bn4a_branch2a', relu=True)
             .conv(3, 3, 256, 1, 1, 2, name='res4a_branch2b', relu=False, biased=False)
             .batch_normalization(name='bn4a_branch2b', relu=True)
             .conv(1, 1, 1024, 1, 1, 1, name='res4a_branch2c', relu=False, biased=False)
             .batch_normalization(name='bn4a_branch2c'))

          (self.feed('bn4a_branch1', 'bn4a_branch2c')
             .add(name='res4a')
             .relu(name='res4a_relu')
             .conv(1, 1, 256, 1, 1, 1, name='res4b_branch2a', relu=False, biased=False)
             .batch_normalization(name='bn4b_branch2a', relu=True)
             .conv(3, 3, 256, 1, 1, 2, name='res4b_branch2b', relu=False, biased=False)
             .batch_normalization(name='bn4b_branch2b', relu=True)
             .conv(1, 1, 1024, 1, 1, 1, name='res4b_branch2c', relu=False, biased=False)
             .batch_normalization(name='bn4b_branch2c'))

          (self.feed('res4a_relu', 'bn4b_branch2c')
             .add(name='res4b')
             .relu(name='res4b_relu')
             .conv(1, 1, 256, 1, 1, 1, name='res4c_branch2a', relu=False, biased=False)
             .batch_normalization(name='bn4c_branch2a', relu=True)
             .conv(3, 3, 256, 1, 1, 2, name='res4c_branch2b', relu=False, biased=False)
             .batch_normalization(name='bn4c_branch2b', relu=True)
             .conv(1, 1, 1024, 1, 1, 1, name='res4c_branch2c', relu=False, biased=False,)
             .batch_normalization(name='bn4c_branch2c'))

          (self.feed('res4b_relu', 'bn4c_branch2c')
             .add(name='res4c')
             .relu(name='res4c_relu')
             .conv(1, 1, 256, 1, 1, 1, name='res4d_branch2a', relu=False, biased=False)
             .batch_normalization(name='bn4d_branch2a', relu=True)
             .conv(3, 3, 256, 1, 1, 2, name='res4d_branch2b', relu=False, biased=False)
             .batch_normalization(name='bn4d_branch2b', relu=True)
             .conv(1, 1, 1024, 1, 1, 1, name='res4d_branch2c', relu=False, biased=False)
             .batch_normalization(name='bn4d_branch2c'))

          (self.feed('res4c_relu', 'bn4d_branch2c')
             .add(name='res4d')
             .relu(name='res4d_relu')
             .conv(1, 1, 256, 1, 1, 1, name='res4e_branch2a', relu=False, biased=False)
             .batch_normalization(name='bn4e_branch2a', relu=True)
             .conv(3, 3, 256, 1, 1, 2, name='res4e_branch2b', relu=False, biased=False)
             .batch_normalization( name='bn4e_branch2b', relu=True)
             .conv(1, 1, 1024, 1, 1, 1, name='res4e_branch2c', relu=False, biased=False)
             .batch_normalization(name='bn4e_branch2c'))

          (self.feed('res4d_relu', 'bn4e_branch2c')
             .add(name='res4e')
             .relu(name='res4e_relu')
             .conv(1, 1, 256, 1, 1, 1, name='res4f_branch2a', relu=False, biased=False)
             .batch_normalization(name='bn4f_branch2a', relu=True)
             .conv(3, 3, 256, 1, 1, 2, name='res4f_branch2b', relu=False, biased=False)
             .batch_normalization(name='bn4f_branch2b', relu=True)
             .conv(1, 1, 1024, 1, 1, 1, name='res4f_branch2c', relu=False, biased=False)
             .batch_normalization(name='bn4f_branch2c'))

          (self.feed('res4e_relu', 'bn4f_branch2c')
             .add(name='res4f')
             .relu(name='res4f_relu'))

          (self.conv(1, 1, 256, 1, 1, 1, name='spp_conv1', relu=False, biased=False)
             .batch_normalization(name='spp_bn1', lrelu=True))

          (self.avg_pool(40, 40, 40, 40, name='spp_pool_a')
             .conv(1, 1, 32, 1, 1, 1, name='spp_conv_a', relu=False, biased=False)
             .batch_normalization(name='spp_bn_a', lrelu=True)
             .resize(40, name='spp_up_a', method=bilinear))

          (self.feed('spp_bn1')
             .avg_pool(20, 20, 20, 20, name='spp_pool_b')
             .conv(1, 1, 32, 1, 1, 1, name='spp_conv_b', relu=False, biased=False)
             .batch_normalization(name='spp_bn_b', lrelu=True)
             .resize(20, name='spp_up_b', method=bilinear))

          (self.feed('spp_bn1')
             .avg_pool(10, 10, 10, 10, name='spp_pool_c')
             .conv(1, 1, 32, 1, 1, 1, name='spp_conv_c', relu=False, biased=False)
             .batch_normalization(name='spp_bn_c', lrelu=True)
             .resize(10, name='spp_up_c', method=bilinear))

          (self.feed('spp_bn1')
             .avg_pool(5, 5, 5, 5, name='spp_pool_d')
             .conv(1, 1, 32, 1, 1, 1, name='spp_conv_d', relu=False, biased=False)
             .batch_normalization(name='spp_bn_d', lrelu=True)
             .resize(5, name='spp_up_d', method=bilinear))

          (self.feed('spp_bn1', 'spp_up_a', 'spp_up_b', 'spp_up_c', 'spp_up_d')
             .concat(axis=3, name='spp_concat')
             .conv(3, 3, 64, 1, 1, 1, name='spp_conv2', relu=False, biased=False)
             .batch_normalization(name='spp_bn2', lrelu=True)
             .resize(2, name='spp_up2', method=bilinear))

          (self.feed('res2c_relu')
             .conv(3, 3, 64, 1, 1, 1, name='spp_conv3_1', relu=False, biased=False)
             .batch_normalization(name='spp_bn3_1', lrelu=True)
             .feed('spp_up2', 'spp_bn3_1')
             .concat(axis=3, name='spp_concat3')
             .conv(3, 3, 64, 1, 1, 1, name='spp_conv3_2', relu=False, biased=False)
             .batch_normalization(name='spp_bn3_2', lrelu=True)
             .resize(2, name='spp_up3', method=bilinear))

          (self.feed('bn_conv1')
             .conv(3, 3, 64, 1, 1, 1, name='spp_conv4_1', relu=False, biased=False)
             .batch_normalization(name='spp_bn4_1', lrelu=True)
             .feed('spp_up3', 'spp_bn4_1')
             .concat(axis=3, name='spp_concat4')
             .conv(3, 3, 64, 1, 1, 1, name='spp_conv4_2', relu=False, biased=False)
             .batch_normalization(name='spp_bn4_2', lrelu=True)
             .resize(2, name='spp_up4', method=bilinear)
             .conv(3, 3, 64, 1, 1, 1, name='depth_conv1', relu=False, biased=False)
             .batch_normalization(name='depth_bn1', lrelu=True)
             .conv(3, 3, 64, 1, 1, 1, name='depth_conv2', relu=False, biased=False)
             .batch_normalization(name='depth_bn2', relu=False)
             .feed('depth_bn1', 'depth_bn2')
             .add(name='depth_add2')
             .leaky_relu(name='depth_relu2')
             .dropout(rate=0.05, name='depth_drop2')
             .conv(1, 1, 1, 1, 1, 1, name='depth_pre', relu=False, biased=True))