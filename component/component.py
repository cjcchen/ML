
import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops


def bn(x, mode, train_ops, bn_epsilon=0.001):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    axis = list(range(len(x_shape) - 1))

    beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer)
    gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer)
    mean, variance = tf.nn.moments(x, axis)

    if mode=='train':
        mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

        moving_mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
        moving_variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)

        train_ops.append(moving_averages.assign_moving_average(
                    moving_mean, mean, 0.9))
        train_ops.append(moving_averages.assign_moving_average(
                    moving_variance, variance, 0.9))
    else:
        mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
        variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)
    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, bn_epsilon)


def conv(x,ksize, output_channel, stride):
    print "conv x:",x.shape, ksize, output_channel, stride
    xshape = [ksize, ksize, x.get_shape()[-1], output_channel]
    #initializer = tf.truncated_normal_initializer(mean=0.0,stddev=0.1)
    initializer = tf.contrib.layers.xavier_initializer( uniform=False )

    with tf.name_scope('conn_weights'):
        weights = tf.get_variable('weights', shape=xshape, dtype='float', initializer=initializer)
        variable_summaries(weights)

    x=tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')
    print "conv done x:",x.shape, ksize, output_channel
    return x

def relu(x):
    return tf.nn.relu(x)

def max_pool(x, ksize, stride):
    return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding='SAME')

def avg_pool(x):
    return tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")

def fc(x, out_d):
    print "fc:",x.shape
    num_units_in = x.get_shape()[1]
    num_units_out =  out_d
    weights_initializer = tf.truncated_normal_initializer(stddev=0.01)

    weights = tf.get_variable('weights', shape=[num_units_in, num_units_out], initializer=weights_initializer)
    biases = tf.get_variable('biases', shape=[num_units_out], initializer=tf.zeros_initializer)
    x = tf.nn.xw_plus_b(x, weights, biases)
    print "fc:",x
    return x

def get_decay(weight_decay_rate=0.0002):
    costs = []
    for var in tf.trainable_variables():
        if var.op.name.find(r'weights') > 0:
            costs.append(tf.nn.l2_loss(var))
    print costs
    return tf.multiply(weight_decay_rate, tf.add_n(costs))

def get_softmax_loss(x,y):
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x, labels=y))
    loss = cross_entropy + get_decay()
    return loss

def variable_summaries(var):
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
    #tf.summary.scalar('mean', mean)

    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

    tf.add_to_collection('stddev',stddev);
    tf.add_to_collection('max',tf.reduce_max(var));
    tf.add_to_collection('min',tf.reduce_min(var));
    tf.add_to_collection('mean',tf.reduce_mean(var));
    tf.add_to_collection('histogram',var);


