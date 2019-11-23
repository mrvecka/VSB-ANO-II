import tensorflow as tf
import config as cfg
# WEIGHTS
def init_weights(shape, name):
    init_random_dist = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(init_random_dist, name=name+'_WEIGHTS')

# BIASES
def init_bias(shape, name):
    init_bias_vals = tf.constant(0.1,shape=shape)
    return tf.Variable(init_bias_vals, name=name+'_BIASES')

# MAX POOL LAYER
def add_max_pool(out_layer, pool_shape, stride, name):
    # perform max pooling
    ksize = [1, pool_shape[0],pool_shape[1],1]
    strides = [1,stride,stride,1]
    out_layer = tf.nn.max_pool(out_layer,ksize=ksize,strides= strides,padding='SAME', name=name)
    return out_layer

# CONVOLUTION LAYER WITH RELU
def convolution_layer_with_relu(input_data, num_input_chanels, num_output_channels, filter_shape, pool_shape,pool_stride, stride, use_maxPool, name):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filter_shape = [filter_shape[0],filter_shape[1], num_input_chanels, num_output_channels]

    # initialize weights anddd bias for the filter
    # weights = tf.Variable(tf.truncated_normal(conv_filter_shape, stddev=0.1),name=name+'_W')
    weights = init_weights(conv_filter_shape, name)

    # bias = tf.constant(0.1,shape=[num_output_channels],name=name+'_b')
    bias = init_bias([num_output_channels],name)

    # setup the convolutional layer network
    out_layer = tf.nn.conv2d(input_data, weights, strides=[1, stride, stride, 1], padding='SAME')

    # add bias
    out_layer = tf.add(out_layer, bias)

    # normalization
    # out_layer = tf.layers.batch_normalization(out_layer,training=is_training)

    # apply a relu non-linea activation
    out_layer = tf.nn.relu(out_layer)
    
    if use_maxPool:
        # perform max pooling
        out_layer = add_max_pool(out_layer, pool_shape, pool_stride, name+"_MAXPOOL")
        
    return out_layer

# FULLY CONNECTED LAYER
def fully_connected_layer(input_layer, size, name):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size,size], name)
    b = init_bias([size], name)
    return tf.add(tf.matmul(input_layer,W), b)

def dropout_layer(input_layer, hold_prob, name):  
    full_dropout = tf.nn.dropout(input_layer,keep_prob=hold_prob, name=name)
    return full_dropout

def normalize_data(data, is_training):
    # normalization
    out = tf.layers.batch_normalization(data, training=is_training)   
    return out

def create_classification_network(image_placeholder, hold_prob, training):
    shaped_image_placeholder = tf.reshape(image_placeholder, [-1, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, cfg.IMAGE_CHANNELS])

    net = convolution_layer_with_relu(shaped_image_placeholder, cfg.IMAGE_CHANNELS, 8, [5, 5], [2, 2], 1, 1, False, name='layer1')
    # net = normalize_data(net, training)
    
    net = convolution_layer_with_relu(net, 8, 16, [5, 5], [2, 2], 2, 1, True, name='layer2')
    # net = normalize_data(net, training)
    
    net = tf.reshape(net, [-1, net.get_shape()[1]*net.get_shape()[2]*net.get_shape()[3]])
    
    net = fully_connected_layer(net, cfg.FULLY_CONNECTED_LAYER_SIZE, "fully_connected_1")
    net = tf.nn.relu(net)
    # net = normalize_data(net, training)

    net = dropout_layer(net, hold_prob, "dropout_1")
    
    y_predicted = fully_connected_layer(net, cfg.NUM_CLASSES, "fully_connected_2")
    
    return y_predicted