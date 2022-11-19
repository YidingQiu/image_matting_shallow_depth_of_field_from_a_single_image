from network.network import Network

# Feature net
class FeaNet(Network):
    def __init__(self, input, is_training, trainable=True, bn_global=None, upsample_size=640):
        # initialization
        self.is_training = is_training
        self.bn_global = bn_global
        self.upsample_size = upsample_size
        super(FeaNet, self).__init__(input, trainable)

    def setup(self):
        # Output image_feature
        (self.feed('image')
        .padding(padding=1, name='image-pad', mode='SYMMETRIC')
        .conv(3, 3, 30, 1, 1, 1, name='fea1', padding='VALID', relu=True, biased=True)
        .padding(padding=2, name='fea1-pad', mode='SYMMETRIC')
        .conv(5, 5, 30, 1, 1, 1, name='fea2', padding='VALID', relu=True, biased=True)
        .padding(padding=2, name='fea2-pad', mode='SYMMETRIC')
        .conv(5, 5, 15, 1, 1, 1, name='fea3', padding='VALID', relu=True, biased=True)
        .padding(padding=2, name='fea3-pad', mode='SYMMETRIC')
        .conv(5, 5, 15, 1, 1, 1, name='fea4', padding='VALID', relu=True, biased=True)
        .feed('image', 'fea1', 'fea2', 'fea3', 'fea4')
        .concat(name='output')
        )
