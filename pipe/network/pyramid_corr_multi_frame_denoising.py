import sys

sys.path.insert(0, './module/')
import tensorflow as tf
from dataset import *
from activation import *
from conv import conv
from dfus_block import dfus_block_add_output_conv
from tensorpack.models import *
from module.utils import bilinear_warp, costvolumelayer

tf.logging.set_verbosity(tf.logging.INFO)

PI = 3.14159265358979323846
flg = False
dtype = tf.float32


def feature_extractor_subnet(x, flg, regular, num=1):
    """Build a U-Net architecture"""

    """ Args: x is the input, 4-D tensor (BxHxWxC)
              flg represent weather add the BN
              regular represent the regularizer number 


        Return: output is 4-D Tensor (BxHxWxC)
    """

    pref = 'feature_extractor_subnet_'+str(num)+'_'

    # whether to train flag
    train_ae = flg

    # define initializer for the network
    keys = ['conv', 'upsample']
    keys_avoid = ['OptimizeLoss']
    inits = []

    init_net = None
    if init_net != None:
        for name in init_net.get_variable_names():
            # select certain variables
            flag_init = False
            for key in keys:
                if key in name:
                    flag_init = True
            for key in keys_avoid:
                if key in name:
                    flag_init = False
            if flag_init:
                name_f = name.replace('/', '_')
                num = str(init_net.get_variable_value(name).tolist())
                exec(
                    "class " + name_f + "(Initializer):\n def __init__(self,dtype=tf.float32): self.dtype=dtype \n def __call__(self,shape,dtype=None,partition_info=None): return tf.cast(np.array(" + num + "),dtype=self.dtype)\n def get_config(self):return {\"dtype\": self.dtype.name}")
                inits.append(name_f)

    # autoencoder
    n_filters = [
        16,
        32,
        64,
        96,
        128,
        192,
    ]
    pool_sizes = [
        1,
        2,
        2,
        2,
        2,
        2,
    ]
    # change space
    ae_inputs = tf.identity(x, name='ae_inputs')

    # prepare input
    current_input = tf.identity(ae_inputs, name="input")

    features = []
    for i in range(0, len(n_filters)):

        if pool_sizes[i] == 1:
            current_input = single_conv(current_input, flg, regular, kernel_size=3, output_channel=n_filters[i],
                                        num=i, previous_pref=pref)
            current_input = tf.nn.relu(current_input)
            current_input = residual_channel_attention_block(current_input, flg, regular, n_filters[i],
                                                             reduction_num=2, subnet_num=i, previous_pref=pref)
            features.append(current_input)
        else:
            current_input = single_conv(current_input, flg, regular, kernel_size=3, output_channel=n_filters[i],
                                        num=i, previous_pref=pref)
            current_input = tf.nn.relu(current_input)
            current_input = tf.layers.max_pooling2d(
                inputs=current_input,
                pool_size=[pool_sizes[i], pool_sizes[i]],
                strides=pool_sizes[i],
                name=pref + "pool_" + str(i)
            )
            current_input = residual_channel_attention_block(current_input, flg, regular, n_filters[i],
                                                             reduction_num=2, subnet_num=i, previous_pref=pref)
            features.append(current_input)
    return features


def depth_residual_regresssion_subnet(x, x_depth, flg, regular, subnet_num):
    """Build a U-Net architecture"""
    """ Args: x is the input, 4-D tensor (BxHxWxC)
              flg represent weather add the BN
              regular represent the regularizer number 


        Return: output is 4-D Tensor (BxHxWxC)
    """

    pref = 'depth_regression_subnet_' + str(subnet_num) + '_'

    # whether to train flag
    train_ae = flg

    # define initializer for the network
    keys = ['conv', 'upsample']
    keys_avoid = ['OptimizeLoss']
    inits = []

    init_net = None
    if init_net != None:
        for name in init_net.get_variable_names():
            # select certain variables
            flag_init = False
            for key in keys:
                if key in name:
                    flag_init = True
            for key in keys_avoid:
                if key in name:
                    flag_init = False
            if flag_init:
                name_f = name.replace('/', '_')
                num = str(init_net.get_variable_value(name).tolist())
                exec(
                    "class " + name_f + "(Initializer):\n def __init__(self,dtype=tf.float32): self.dtype=dtype \n def __call__(self,shape,dtype=None,partition_info=None): return tf.cast(np.array(" + num + "),dtype=self.dtype)\n def get_config(self):return {\"dtype\": self.dtype.name}")
                inits.append(name_f)

    # autoencoder
    n_filters = [
        128, 96,
        64, 32,
        16, 1,
    ]
    filter_sizes = [
        3, 3,
        3, 3,
        3, 3,
    ]
    pool_sizes = [ \
        1, 1,
        1, 1,
        1, 1,
    ]
    pool_strides = [
        1, 1,
        1, 1,
        1, 1,
    ]
    skips = [ \
        False, False,
        False, False,
        False, False,
    ]
    # change space
    ae_inputs = tf.identity(x, name='ae_inputs')

    # prepare input
    current_input = tf.identity(ae_inputs, name="input")
    ####################################################################################################################
    # convolutional layers: depth regression
    feature = []
    for i in range(0, len(n_filters)):
        name = pref + "conv_" + str(i)

        # define the initializer
        if name + '_bias' in inits:
            bias_init = eval(name + '_bias()')
        else:
            bias_init = tf.zeros_initializer()
        if name + '_kernel' in inits:
            kernel_init = eval(name + '_kernel()')
        else:
            kernel_init = None
        if i == (len(n_filters) - 1):
            activation = None
        else:
            activation = relu

        # convolution

        current_input = tf.layers.conv2d(
            inputs=current_input,
            filters=n_filters[i],
            kernel_size=[filter_sizes[i], filter_sizes[i]],
            padding="same",
            activation=activation,
            trainable=train_ae,
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            name=name,
        )

        feature.append(current_input)
        current_input = feature[-1]

    depth_coarse = tf.identity(feature[-1], name='depth_coarse_output')
    return depth_coarse

def corr_feature_regression_subnet(x, flg, regular, subnet_num):
    """Build a U-Net architecture"""
    """ Args: x is the input, 4-D tensor (BxHxWxC)
              flg represent weather add the BN
              regular represent the regularizer number 


        Return: output is 4-D Tensor (BxHxWxC)
    """

    pref = 'corr_feature_regression_subnet_' + str(subnet_num) + '_'

    # whether to train flag
    train_ae = flg

    # define initializer for the network
    keys = ['conv', 'upsample']
    keys_avoid = ['OptimizeLoss']
    inits = []

    init_net = None
    if init_net != None:
        for name in init_net.get_variable_names():
            # select certain variables
            flag_init = False
            for key in keys:
                if key in name:
                    flag_init = True
            for key in keys_avoid:
                if key in name:
                    flag_init = False
            if flag_init:
                name_f = name.replace('/', '_')
                num = str(init_net.get_variable_value(name).tolist())
                exec(
                    "class " + name_f + "(Initializer):\n def __init__(self,dtype=tf.float32): self.dtype=dtype \n def __call__(self,shape,dtype=None,partition_info=None): return tf.cast(np.array(" + num + "),dtype=self.dtype)\n def get_config(self):return {\"dtype\": self.dtype.name}")
                inits.append(name_f)

    # autoencoder
    n_filters = [
        32, 16,
    ]
    filter_sizes = [
        3, 3,
    ]
    pool_sizes = [ \
        1, 1,
    ]
    pool_strides = [
        1, 1,
    ]
    skips = [ \
        False, False,
    ]
    # change space
    ae_inputs = tf.identity(x, name='ae_inputs')

    # prepare input
    current_input = tf.identity(ae_inputs, name="input")
    ####################################################################################################################
    # convolutional layers: depth regression
    feature = []
    for i in range(0, len(n_filters)):
        name = pref + "conv_" + str(i)

        # define the initializer
        if name + '_bias' in inits:
            bias_init = eval(name + '_bias()')
        else:
            bias_init = tf.zeros_initializer()
        if name + '_kernel' in inits:
            kernel_init = eval(name + '_kernel()')
        else:
            kernel_init = None
        if i == (len(n_filters) - 1):
            activation = None
        else:
            activation = relu

        # convolution

        current_input = tf.layers.conv2d(
            inputs=current_input,
            filters=n_filters[i],
            kernel_size=[filter_sizes[i], filter_sizes[i]],
            padding="same",
            activation=activation,
            trainable=train_ae,
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            name=name,
        )

        feature.append(current_input)
        current_input = feature[-1]

    depth_coarse = tf.identity(feature[-1], name='depth_coarse_output')
    return depth_coarse

def unet_subnet(x, flg, regular):
    """Build a U-Net architecture"""

    """ Args: x is the input, 4-D tensor (BxHxWxC)
              flg represent weather add the BN
              regular represent the regularizer number 


        Return: output is 4-D Tensor (BxHxWxC)
    """

    pref = 'unet_subnet_'

    # whether to train flag
    train_ae = flg

    # define initializer for the network
    keys = ['conv', 'upsample']
    keys_avoid = ['OptimizeLoss']
    inits = []

    init_net = None
    if init_net != None:
        for name in init_net.get_variable_names():
            # select certain variables
            flag_init = False
            for key in keys:
                if key in name:
                    flag_init = True
            for key in keys_avoid:
                if key in name:
                    flag_init = False
            if flag_init:
                name_f = name.replace('/', '_')
                num = str(init_net.get_variable_value(name).tolist())
                # self define the initializer function
                from tensorflow.python.framework import dtypes
                from tensorflow.python.ops.init_ops import Initializer
                exec(
                    "class " + name_f + "(Initializer):\n def __init__(self,dtype=tf.float32): self.dtype=dtype \n def __call__(self,shape,dtype=None,partition_info=None): return tf.cast(np.array(" + num + "),dtype=self.dtype)\n def get_config(self):return {\"dtype\": self.dtype.name}")
                inits.append(name_f)

    # autoencoder
    n_filters = [
        16, 16,
        32, 32,
        64, 64,
        128, 128,
    ]
    filter_sizes = [
        3, 3,
        3, 3,
        3, 3,
        3, 3,
    ]
    pool_sizes = [ \
        1, 1,
        2, 1,
        2, 1,
        2, 1,
    ]
    pool_strides = [
        1, 1,
        2, 1,
        2, 1,
        2, 1,
    ]
    skips = [ \
        False, False,
        True, False,
        True, False,
        True, False,
    ]
    # change space
    ae_inputs = tf.identity(x, name='ae_inputs')

    # prepare input
    current_input = tf.identity(ae_inputs, name="input")
    ####################################################################################################################
    # convolutional layers: encoder
    conv = []
    pool = [current_input]
    for i in range(0, len(n_filters)):
        name = pref + "conv_" + str(i)

        # define the initializer
        if name + '_bias' in inits:
            bias_init = eval(name + '_bias()')
        else:
            bias_init = tf.zeros_initializer()
        if name + '_kernel' in inits:
            kernel_init = eval(name + '_kernel()')
        else:
            kernel_init = None

        # convolution
        conv.append( \
            tf.layers.conv2d( \
                inputs=current_input,
                filters=n_filters[i],
                kernel_size=[filter_sizes[i], filter_sizes[i]],
                padding="same",
                activation=relu,
                trainable=train_ae,
                kernel_initializer=kernel_init,
                bias_initializer=bias_init,
                name=name,
            )
        )
        if pool_sizes[i] == 1 and pool_strides[i] == 1:
            pool.append(conv[-1])
        else:
            pool.append( \
                tf.layers.max_pooling2d( \
                    inputs=conv[-1],
                    pool_size=[pool_sizes[i], pool_sizes[i]],
                    strides=pool_strides[i],
                    name=pref + "pool_" + str(i)
                )
            )
        current_input = pool[-1]
    ####################################################################################################################
    # convolutional layer: decoder
    # upsampling
    upsamp = []
    current_input = pool[-1]
    for i in range((len(n_filters) - 1) - 1, 0, -1):
        name = pref + "upsample_" + str(i)

        # define the initializer
        if name + '_bias' in inits:
            bias_init = eval(name + '_bias()')
        else:
            bias_init = tf.zeros_initializer()
        if name + '_kernel' in inits:
            kernel_init = eval(name + '_kernel()')
        else:
            kernel_init = None
        ## change the kernel size in upsample process
        if skips[i] == False and skips[i + 1] == True:
            filter_sizes[i] = 4
        # upsampling
        current_input = tf.layers.conv2d_transpose( \
            inputs=current_input,
            filters=n_filters[i],
            kernel_size=[filter_sizes[i], filter_sizes[i]],
            strides=(pool_strides[i], pool_strides[i]),
            padding="same",
            activation=relu,
            trainable=train_ae,
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            name=name
        )
        upsamp.append(current_input)
        # current_input = tf.layers.batch_normalization(
        #     inputs=current_input,
        #     training=train_ae,
        #     name=pref + "upsamp_BN_" + str(i))
        # skip connection
        if skips[i] == False and skips[i - 1] == True:
            current_input = tf.concat([current_input, pool[i + 1]], axis=-1)
    ####################################################################################################################
    features = tf.identity(upsamp[-1], name='ae_output')
    return features

def depth_output_subnet(inputs, flg, regular, kernel_size):  ## x (B,H,W,1), features:(B,H,W,64), samples:(B,H,W,9)
    pref = 'depth_output_subnet_'

    # whether to train flag
    train_ae = flg
    current_input = inputs
    # define initializer for the network
    keys = ['conv', 'upsample']
    keys_avoid = ['OptimizeLoss']
    inits = []

    init_net = None
    if init_net != None:
        for name in init_net.get_variable_names():
            # select certain variables
            flag_init = False
            for key in keys:
                if key in name:
                    flag_init = True
            for key in keys_avoid:
                if key in name:
                    flag_init = False
            if flag_init:
                name_f = name.replace('/', '_')
                num = str(init_net.get_variable_value(name).tolist())
                # self define the initializer function
                from tensorflow.python.framework import dtypes
                from tensorflow.python.ops.init_ops import Initializer
                exec(
                    "class " + name_f + "(Initializer):\n def __init__(self,dtype=tf.float32): self.dtype=dtype \n def __call__(self,shape,dtype=None,partition_info=None): return tf.cast(np.array(" + num + "),dtype=self.dtype)\n def get_config(self):return {\"dtype\": self.dtype.name}")
                inits.append(name_f)

    n_filters_mix = [kernel_size ** 2]
    filter_sizes_mix = [1]
    mix = []
    for i in range(len(n_filters_mix)):
        name = pref + "conv_" + str(i)

        # define the initializer
        if name + '_bias' in inits:
            bias_init = eval(name + '_bias()')
        else:
            bias_init = tf.zeros_initializer()
        if name + '_kernel' in inits:
            kernel_init = eval(name + '_kernel()')
        else:
            kernel_init = None

        if i == (len(n_filters_mix) - 1):
            activation = sigmoid
        else:
            activation = relu

        # convolution
        mix.append( \
            tf.layers.conv2d( \
                inputs=current_input,
                filters=n_filters_mix[i],
                kernel_size=[filter_sizes_mix[i], filter_sizes_mix[i]],
                padding="same",
                activation=activation,
                trainable=train_ae,
                kernel_initializer=kernel_init,
                bias_initializer=bias_init,
                name=name,
            )
        )
        current_input = mix[-1]

    # biases = current_input[:, :, :, 0::0 - kernel_size ** 2]
    # weights = current_input[:, :, :, 0 - kernel_size ** 2::]
    ## run y = w(x + b)

    return current_input

def dear_kpn(x, flg, regular):

    kernel_size = 3
    features = unet_subnet(x, flg, regular)
    print(features.shape.as_list())
    weights = depth_output_subnet(features, flg, regular, kernel_size=kernel_size)
    weights = weights / tf.reduce_sum(tf.abs(weights) + 1e-6, axis=-1, keep_dims=True)
    print(weights.shape.as_list())
    print(x.shape.as_list())
    column = im2col(x, kernel_size=kernel_size)
    current_output = tf.reduce_sum(column * weights, axis=-1, keep_dims=True)
    depth_output = tf.identity(current_output, name='depth_output')

    return depth_output

def residual_output_subnet(x, flg, regular, subnet_num):
    """Build a U-Net architecture"""
    """ Args: x is the input, 4-D tensor (BxHxWxC)
              flg represent weather add the BN
              regular represent the regularizer number 


        Return: output is 4-D Tensor (BxHxWxC)
    """

    pref = 'residual_output_subnet_' + str(subnet_num) + '_'

    # whether to train flag
    train_ae = flg

    # define initializer for the network
    keys = ['conv', 'upsample']
    keys_avoid = ['OptimizeLoss']
    inits = []

    init_net = None
    if init_net != None:
        for name in init_net.get_variable_names():
            # select certain variables
            flag_init = False
            for key in keys:
                if key in name:
                    flag_init = True
            for key in keys_avoid:
                if key in name:
                    flag_init = False
            if flag_init:
                name_f = name.replace('/', '_')
                num = str(init_net.get_variable_value(name).tolist())
                # self define the initializer function
                from tensorflow.python.framework import dtypes
                from tensorflow.python.ops.init_ops import Initializer
                exec(
                    "class " + name_f + "(Initializer):\n def __init__(self,dtype=tf.float32): self.dtype=dtype \n def __call__(self,shape,dtype=None,partition_info=None): return tf.cast(np.array(" + num + "),dtype=self.dtype)\n def get_config(self):return {\"dtype\": self.dtype.name}")
                inits.append(name_f)

    # autoencoder
    n_filters = [
        1
    ]
    filter_sizes = [
        1
    ]
    pool_sizes = [ \
        1
    ]
    pool_strides = [
        1
    ]
    skips = [ \
        False
    ]
    # change space
    ae_inputs = tf.identity(x, name='ae_inputs')

    # prepare input
    current_input = tf.identity(ae_inputs, name="input")
    ####################################################################################################################
    # convolutional layers: depth regression
    feature = []
    for i in range(0, len(n_filters)):
        name = pref + "conv_" + str(i)

        # define the initializer
        if name + '_bias' in inits:
            bias_init = eval(name + '_bias()')
        else:
            bias_init = tf.zeros_initializer()
        if name + '_kernel' in inits:
            kernel_init = eval(name + '_kernel()')
        else:
            kernel_init = None
        if i == (len(n_filters) - 1):
            activation = None
        else:
            activation = relu

        # convolution
        current_input = tf.layers.conv2d(
            inputs=current_input,
            filters=n_filters[i],
            kernel_size=[filter_sizes[i], filter_sizes[i]],
            padding="same",
            activation=activation,
            trainable=train_ae,
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            name=name,
        )

        if pool_sizes[i] == 1 and pool_strides[i] == 1:
            feature.append(current_input)
        else:
            feature.append(
                tf.layers.max_pooling2d( \
                    inputs=current_input,
                    pool_size=[pool_sizes[i], pool_sizes[i]],
                    strides=pool_strides[i],
                    name=pref + "pool_" + str(i)
                )
            )
        current_input = feature[-1]

    depth_residual_coarse = tf.identity(feature[-1], name='depth_coarse_residual_output')
    return depth_residual_coarse

def single_conv(x, flg, regular, kernel_size, output_channel, num, previous_pref):
    """ Args: x is the input, 4-D tensor (BxHxWxC)
              flg represent weather add the BN
              regular represent the regularizer number
        Return: output is 4-D Tensor (BxHxWxC)
    """

    pref = previous_pref + 'single_conv_' + str(num) + '_'
    train_ae = flg
    keys = ['conv', 'upsample']
    keys_avoid = ['OptimizeLoss']
    inits = []

    init_net = None
    if init_net != None:
        for name in init_net.get_variable_names():
            # select certain variables
            flag_init = False
            for key in keys:
                if key in name:
                    flag_init = True
            for key in keys_avoid:
                if key in name:
                    flag_init = False
            if flag_init:
                name_f = name.replace('/', '_')
                num = str(init_net.get_variable_value(name).tolist())
                exec(
                    "class " + name_f + "(Initializer):\n def __init__(self,dtype=tf.float32): self.dtype=dtype \n def __call__(self,shape,dtype=None,partition_info=None): return tf.cast(np.array(" + num + "),dtype=self.dtype)\n def get_config(self):return {\"dtype\": self.dtype.name}")
                inits.append(name_f)
    n_filters = [output_channel]
    filter_sizes = [kernel_size]
    ae_inputs = tf.identity(x, name='ae_inputs')
    current_input = tf.identity(ae_inputs, name="input")
    for i in range(0, len(n_filters)):
        name = pref + "conv_" + str(i)
        if name + '_bias' in inits:
            bias_init = eval(name + '_bias()')
        else:
            bias_init = tf.zeros_initializer()
        if name + '_kernel' in inits:
            kernel_init = eval(name + '_kernel()')
        else:
            kernel_init = None
        activation = None

        current_input = tf.layers.conv2d(
            inputs=current_input,
            filters=n_filters[i],
            kernel_size=[filter_sizes[i], filter_sizes[i]],
            padding="same",
            activation=activation,
            trainable=train_ae,
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            name=name,
            reuse=tf.AUTO_REUSE
        )
    single_conv_output = tf.identity(current_input, name='single_conv_output_' + str(num))
    return single_conv_output

def residual_channel_attention_block(x, flg, regular, in_channel, reduction_num, subnet_num, previous_pref):

    pref = previous_pref + 'recsidual_channel_attention_block_' + str(subnet_num) + '_'

    f = single_conv(x, flg, regular, 3, in_channel, 1, pref)
    f = tf.nn.relu(f)
    f = single_conv(f, flg, regular, 3, in_channel, 2, pref)

    y = tf.reduce_mean(f, axis=(1, 2), keepdims=True)  # (B, 1, 1, C)
    y = single_conv(y, flg, regular, 1, in_channel // reduction_num, 3, pref)
    y = tf.nn.relu(y)
    y = single_conv(y, flg, regular, 1, in_channel, 4, pref)
    y = tf.nn.sigmoid(y)
    y = tf.multiply(f, y)

    return tf.add(x, y)

def mask_regression_subnet(x, flg, regular, subnet_num):
    """Build a U-Net architecture"""
    """ Args: x is the input, 4-D tensor (BxHxWxC)
              flg represent weather add the BN
              regular represent the regularizer number 


        Return: output is 4-D Tensor (BxHxWxC)
    """

    pref = 'mask_regression_subnet_' + str(subnet_num) + '_'

    # whether to train flag
    train_ae = flg

    # define initializer for the network
    keys = ['conv', 'upsample']
    keys_avoid = ['OptimizeLoss']
    inits = []

    init_net = None
    if init_net != None:
        for name in init_net.get_variable_names():
            # select certain variables
            flag_init = False
            for key in keys:
                if key in name:
                    flag_init = True
            for key in keys_avoid:
                if key in name:
                    flag_init = False
            if flag_init:
                name_f = name.replace('/', '_')
                num = str(init_net.get_variable_value(name).tolist())
                exec(
                    "class " + name_f + "(Initializer):\n def __init__(self,dtype=tf.float32): self.dtype=dtype \n def __call__(self,shape,dtype=None,partition_info=None): return tf.cast(np.array(" + num + "),dtype=self.dtype)\n def get_config(self):return {\"dtype\": self.dtype.name}")
                inits.append(name_f)

    # autoencoder
    n_filters = [
        32, 1,
    ]
    filter_sizes = [
        3, 3,
    ]
    pool_sizes = [ \
        1, 1,
    ]
    pool_strides = [
        1, 1,
    ]
    skips = [ \
        False, False,
    ]
    # change space
    ae_inputs = tf.identity(x, name='ae_inputs')

    # prepare input
    current_input = tf.identity(ae_inputs, name="input")
    ####################################################################################################################
    # convolutional layers: depth regression
    feature = []
    for i in range(0, len(n_filters)):
        name = pref + "conv_" + str(i)

        # define the initializer
        if name + '_bias' in inits:
            bias_init = eval(name + '_bias()')
        else:
            bias_init = tf.zeros_initializer()
        if name + '_kernel' in inits:
            kernel_init = eval(name + '_kernel()')
        else:
            kernel_init = None
        if i == (len(n_filters) - 1):
            activation = None
        else:
            activation = relu

        # convolution

        current_input = tf.layers.conv2d(
            inputs=current_input,
            filters=n_filters[i],
            kernel_size=[filter_sizes[i], filter_sizes[i]],
            padding="same",
            activation=activation,
            trainable=train_ae,
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            name=name,
        )

        feature.append(current_input)
        current_input = feature[-1]

    output = feature[-1]
    depth_coarse = tf.identity(output, name='depth_coarse_output')
    return depth_coarse

def pyramid_corr_mask_multi_frame_denoising(x, flg, regular, batch_size, deformable_range):

    depth_residual = []
    depth_residual_input = []
    d_conf_list = []
    corr_feature_list = []
    corr_feature_in_list = []
    depth_refine = []
    offsets_scale = []
    depth_residual_weight = [0.32, 0.08, 0.02, 0.01, 0.005]

    h_max = tf.shape(x)[1]
    w_max = tf.shape(x)[2]

    # HAMMER dataset
    depth = tf.expand_dims(x[:, :, :, 0], axis=-1)
    amplitude = tf.expand_dims(x[:, :, :, 1], axis=-1)
    depth_2 = tf.expand_dims(x[:, :, :, 2], axis=-1)
    amplitude_2 = tf.expand_dims(x[:, :, :, 3], axis=-1)
    x1_input = tf.concat([depth, amplitude], axis=-1)
    x2_input = tf.concat([depth_2, amplitude_2], axis=-1)
    features1 = feature_extractor_subnet(x1_input, flg, regular, num=1)
    features2 = feature_extractor_subnet(x2_input, flg, regular, num=1)

    low_num = 2
    for i in range(1, len(features1) + 1):

        if i == 1:
            cost = costvolumelayer(features1[len(features1) - i], features2[len(features1) - i], search_range=3)
            cost_in = costvolumelayer(features1[len(features1) - i], features2[len(features1) - i], search_range=3)
            feature_input = features1[len(features1) - i]
            feature_input_2 = features2[len(features1) - i]
            inputs = tf.concat([feature_input, cost_in],axis=-1)
            m_inputs = tf.concat([feature_input, feature_input_2, cost], axis=-1)
        else:
            feature_input = features1[len(features1) - i]
            feature_input_2 = features2[len(features1) - i]
            h_max_low_scale = tf.shape(feature_input)[1]
            w_max_low_scale = tf.shape(feature_input)[2]
            depth_coarse_input = tf.image.resize_bicubic(depth_residual[-1], size=(h_max_low_scale, w_max_low_scale),
                                                         align_corners=True)
            d_conf = tf.image.resize_bicubic(d_conf_list[-1], size=(h_max_low_scale, w_max_low_scale),
                                             align_corners=True)
            corr_feature = tf.image.resize_bicubic(corr_feature_list[-1], size=(h_max_low_scale, w_max_low_scale),
                                           align_corners=True)
            corr_feature_in = tf.image.resize_bicubic(corr_feature_in_list[-1], size=(h_max_low_scale, w_max_low_scale),
                                                   align_corners=True)
            if i < low_num:
                cost = costvolumelayer(features1[len(features1) - i], features2[len(features2) - i], search_range=3)
                cost_in = costvolumelayer(features1[len(features1) - i], features2[len(features1) - i], search_range=3)
            m_inputs = tf.concat([feature_input, feature_input_2, corr_feature, d_conf], axis=-1)
            inputs = tf.concat([feature_input, corr_feature_in, depth_coarse_input], axis=-1)

        if i < low_num:
            corr_feature = corr_feature_regression_subnet(m_inputs, flg, regular, subnet_num=i)
            corr_feature_in = corr_feature_regression_subnet(inputs, flg, regular, subnet_num=i+1)
            corr_feature_list.append(corr_feature)
            corr_feature_in_list.append(corr_feature_in)
        d_conf = mask_regression_subnet(m_inputs, flg, regular, subnet_num=i)
        d_conf = tf.nn.softmax(d_conf)
        d_conf_list.append(d_conf)
        inputs = tf.concat([inputs, corr_feature], axis=-1)
        current_depth_residual = depth_residual_regresssion_subnet(inputs, depth, flg, regular, subnet_num=i)
        current_depth_residual = current_depth_residual * d_conf
        depth_residual.append(current_depth_residual)
        current_depth_residual_input = tf.image.resize_bicubic(current_depth_residual, size=(h_max, w_max),
                                                               align_corners=True)
        depth_residual_input.append(current_depth_residual_input)

    depth_coarse_residual_input = tf.concat(depth_residual_input, axis=-1)

    final_depth_residual_output = residual_output_subnet(depth_coarse_residual_input, flg, regular, subnet_num=0)

    current_final_depth_output = depth + final_depth_residual_output


    final_depth_output = dear_kpn(current_final_depth_output, flg, regular)
    depth_residual_input.append(final_depth_residual_output)
    depth_residual_input.append(final_depth_output - current_final_depth_output)
    return final_depth_output, tf.concat(depth_residual_input, axis=-1)