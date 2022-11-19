import numpy as np
import tensorflow as tf

# Default padding mode and variance
DEFAULT_PADDING = 'SAME'
DEFAULT_INIT = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")

# Build layer
def layer(operation):
    # Build composable network layer
    def layer_builder(self, *args, **kwargs):
        # Set unique name if not provided
        name = kwargs.setdefault('name', self.default_name(operation.__name__))

        # Add inputs according to the length of terminals
        length = len(self.terminals)
        input = list(self.terminals) if length != 1 else self.terminals[0]

        # Implement operation and get output
        output = operation(self, input, *args, **kwargs)

        # Set output to corresponding operation
        self.layers[name] = output

        # Add output of this layer to the next layer
        self.feed(output)

        # Chained calls
        return self

    return layer_builder


class Network(object):
    # initialization
    def __init__(self, inputs, trainable=True):
        # Inputs for this network
        self.inputs = inputs

        # List of terminal nodes
        self.terminals = []

        # Mapping
        self.layers = dict(inputs)

        # Set variables as trainable if true, initialize regularizer
        self.trainable = trainable
        self.regularizer = [0.0]
        self.setup()

    def setup(self):
        print("Could not implement without subclass!")
    
    # Feed the input(s) for the next operation
    def feed(self, *args):
        # Start of a new layer, reset the terminals
        self.terminals = []

        # For each layer feed the output to the next layer
        for arg in args:
            # Check key string
            if isinstance(arg, str):
                arg = self.layers[arg]
            self.terminals.append(arg)

        return self
    
    # Get current output of the network
    def curr_node(self):
        return self.terminals[-1]

    # Auto increase number which added to the last position
    def default_name(self, name):
        idx = sum(layer_name.startswith(name) for layer_name, _ in self.layers.items()) + 1
        return '%s_%d' % (name, idx)
    
    # Create a new variable
    def var(self, name, shape, lam=0, trainable=None, initializer=None):
        # If not provided trainable, set it the same as others in this layer
        if trainable is None:
            trainable = self.trainable

        # Create variance   
        variance = tf.compat.v1.get_variable(name, shape, trainable=trainable, initializer=initializer)
        if not lam:
            # Add regularizaer as lambda * L2 loss
            self.regularizer.append(lam * tf.nn.l2_loss(variance))

        return variance

    @layer
    # Convolution
    def conv(self, input, kernel_h, kernel_w, channel_output, stride_h, stride_w, dilation_rate, name, lam=5e-4, relu=True, padding=DEFAULT_PADDING, group=1, biased=True):

        # Get the number of channels in the input
        channel_input = input.get_shape()[-1]

        # Define convolution for a given input and kernel with dilation_rate
        if dilation_rate == 1:
            s = [1, stride_h, stride_w, 1]
            # Simple convolution
            conv = lambda i, f: tf.nn.conv2d(input=i, filters=f, strides=s, padding=padding)
        else:
            # Atrous convolution
            conv = lambda i, f: tf.nn.atrous_conv2d(i, f, dilation_rate, padding=padding)

        with tf.compat.v1.variable_scope(name) as scope:
            # Create kernel
            kernel = self.var('weights', shape=[kernel_h, kernel_w, int(channel_input / group), channel_output], lam=lam, initializer=DEFAULT_INIT)
            if group == 1:
                # Simple convolution
                output = conv(input, kernel)
            else:
                # Split the input into groups and then convolve each of them independently
                output_g = [conv(i, f) for i, f in zip(tf.split(input, group, axis=3), tf.split(kernel, group, axis=3))]

                # Concatenate the groups
                output = tf.concat(output_g, 3)

            # Add the biases
            if biased:
                output = tf.nn.bias_add(output, self.var('biases', [channel_output]))
            if relu:
                # Apply ReLU
                output = tf.nn.relu(output, name=scope.name)

            return output

    @layer
    # seperable convolution
    def separable_conv(self, input, kernel_h, kernel_w, channel_output, stride_h, stride_w, dilation_rate, name, channel_middle = 1, lam=5e-4, relu=True, padding=DEFAULT_PADDING, biased=True):

        # Get the number of channels in the input
        channel_input = input.get_shape()[-1]

        # Depthwise and pointwise convolution
        s = [1, stride_h, stride_w, 1]
        separable_conv = lambda i, f_depth, f_point: tf.nn.separable_conv2d(input=i, depthwise_filter=f_depth, pointwise_filter=f_point, strides=s, dilations=[dilation_rate, dilation_rate], padding=padding, name=name)
        with tf.compat.v1.variable_scope(name) as scope:
            # Define depth-wise kernel
            d_shape = [kernel_h, kernel_w, channel_input, channel_middle]
            f_depth = self.var('kernel_depth', shape=d_shape, lam=lam, initializer=DEFAULT_INIT)
            # Define point-wise kernel
            p_shape = [1, 1, channel_input*channel_middle, channel_output]
            f_point = self.var('kernel_point', shape=p_shape, lam=lam, initializer=DEFAULT_INIT)
            # Apply separable convolution
            output = separable_conv(input, f_depth, f_point)

            # Add the biases
            if biased:
                output = tf.nn.bias_add(output, self.var('biases', [channel_output]))
            if relu:
                # Apply ReLU
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    # deconvolution
    def deconv(self, input, kernel_h, kernel_w, channel_output, stride_h, stride_w, name, relu=True, padding=DEFAULT_PADDING, trainable=None, group=1, biased=True):

        # Get the number of channels in the input
        n_input, h_input, w_input, channel_input = input.get_shape().as_list()

        if padding == 'VALID':
            pad_h = pad_w = np.int(0)
        else:
            pad_h = np.int(np.floor((kernel_h - stride_h) / 2.0))
            pad_w = np.int(np.floor((kernel_w - stride_w) / 2.0))

        # calculte the output h and w
        h_output = (h_input - 1) * stride_h + kernel_h - 2 * pad_h
        w_output = (w_input - 1) * stride_w + kernel_w - 2 * pad_w
        
        # Define convolution for a given input and kernel
        deconvolve = lambda i, f: tf.nn.conv2d_transpose(i, f, output_shape=[n_input, h_output, w_output, channel_output/group], strides=[1, stride_h, stride_w, 1], padding=padding)
        with tf.compat.v1.variable_scope(name) as scope:
            kernel = self.var('weights', shape=[kernel_h, kernel_w, channel_input / group, channel_output], trainable=trainable)

            if group == 1:
                # Simple convolution
                output = deconvolve(input, kernel)
            else:
                # Split the input into groups and then convolve each of them independently
                output_groups = [deconvolve(i, f) for i, f in zip(tf.split(input, group, axis=3), tf.split(kernel, group, axis=3))]

                # Concatenate the groups
                output = tf.concat(output_groups, 3)

            # Add the biases
            if biased:
                output = tf.nn.bias_add(output, self.var('biases', [channel_output], trainable=trainable))
            if relu:
                # Apply ReLU
                output = tf.nn.relu(output, name=scope.name)

            return output

    @layer
    # ReLU
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    # Max pooling
    def max_pool(self, input, kernel_h, kernel_w, stride_h, stride_w, name, padding=DEFAULT_PADDING):
        k, s = [1, kernel_h, kernel_w, 1], [1, stride_h, stride_w, 1]
        return tf.nn.max_pool2d(input=input, ksize=k, strides=s, padding=padding, name=name)

    @layer
    # Average pooling
    def avg_pool(self, input, kernel_h, kernel_w, stride_h, stride_w, name, padding=DEFAULT_PADDING):
        k, s = [1, kernel_h, kernel_w, 1], [1, stride_h, stride_w, 1]
        return tf.nn.avg_pool2d(input=input, ksize=k, strides=s, padding=padding, name=name)

    @layer
    # Concate tensor
    def concat(self, inputs, axis=3, name=None):
        return tf.concat(values=inputs, axis=axis, name=name)

    @layer
    # Add tensor together such as shortcut
    def add(self, inputs, name):
        return tf.add_n(inputs, name=name)

    @layer
    # Batch normalization
    def batch_normalization(self, input, name, is_training=None, scale_offset=True, relu=False, lrelu=False, alpha=0.2, momentum=0.99, renorm=False, epsilon=1e-5):
        # Set default training as others 
        if is_training is None:
            is_training = self.is_training

        output = tf.compat.v1.layers.batch_normalization(input, training=is_training, name=name, center=scale_offset, scale=scale_offset, momentum=momentum, epsilon=epsilon, renorm=renorm)
        if relu:
            with tf.compat.v1.variable_scope(name):
                output = tf.nn.relu(output)
        elif lrelu:
            with tf.compat.v1.variable_scope(name):
                output = tf.nn.leaky_relu(output, alpha=alpha)

        return output

    @layer
    # Dropout layer
    def dropout(self, input, rate, name):
        return tf.compat.v1.layers.dropout(input, rate=rate, training=self.is_training, name=name)

    @layer
    # Padding
    def padding(self, input, padding, name, mode='CONSTANT'):
        p = [[0,0], [padding, padding], [padding, padding], [0,0]]
        return tf.pad(tensor=input, paddings=p, mode=mode, name=name)

    @layer
    # Resize the tensor
    def resize(self, input, factor=None, shape=None, name=None, method=tf.image.ResizeMethod.BILINEAR):
        # If provided factore, multiply it by the shape of input
        if factor is not None:
            o_shape = tf.shape(input)[1:3] * factor
        # If provided with shape, set the shape of output as the shape
        elif shape is not None:
            o_shape = shape

        return tf.image.resize(images = input, size = o_shape, method=method)

    @layer
    # Leaky_relu activation
    def leaky_relu(self, input, alpha=0.2, name=None):
        return tf.nn.leaky_relu(input, alpha=alpha, name=name)


