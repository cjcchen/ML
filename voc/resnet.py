import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
import datetime

weight_decay_rate=0.002
BN_EPSILON = 0.001

UPDATE_OPS_COLLECTION='resnet_update_ops'
is_training = tf.placeholder('bool', [], name='is_training')
train_ops = []

def get_variable(name, shape, initializer, dtype='float', trainable=True):
    return tf.get_variable(name, shape=shape, initializer=initializer, trainable=trainable)

def variable_summaries(var):
    with tf.name_scope('summaries'):
	mean = tf.reduce_mean(var)
    #tf.summary.scalar('mean', mean)

    with tf.name_scope('stddev'):
	stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

    #tf.summary.scalar('stddev', stddev)
    #tf.summary.scalar('max', tf.reduce_max(var))
    #tf.summary.scalar('min', tf.reduce_min(var))
    #print var
    #tf.summary.histogram('histogram', var)
    tf.add_to_collection('stddev',stddev);
    tf.add_to_collection('max',tf.reduce_max(var));
    tf.add_to_collection('min',tf.reduce_min(var));
    tf.add_to_collection('mean',tf.reduce_mean(var));
    tf.add_to_collection('histogram',var);

'''
def bn(x):

    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    axis = list(range(len(x_shape) - 1))

    beta = get_variable('beta', params_shape, initializer=tf.zeros_initializer)
    gamma = get_variable('gamma', params_shape, initializer=tf.ones_initializer)
    mean, variance = tf.nn.moments(x, axis)

    moving_mean = get_variable('moving_mean', params_shape, initializer=tf.zeros_initializer, trainable=False)
    moving_variance = get_variable('moving_variance', params_shape, initializer=tf.ones_initializer, trainable=False)
    train_ops.append(moving_averages.assign_moving_average( moving_mean, mean, 0.9))
    train_ops.append(moving_averages.assign_moving_average( moving_variance, variance, 0.9))

    mean, variance = control_flow_ops.cond( is_training, lambda: (mean, variance), lambda: (moving_mean, moving_variance))

    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
'''

def bn(x):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    axis = list(range(len(x_shape) - 1))

    beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer)
    gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer)
    mean, variance = tf.nn.moments(x, axis)

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


    #moving_mean = tf.get_variable('moving_mean', params_shape, initializer=tf.zeros_initializer, trainable=False)
    #moving_variance = tf.get_variable('moving_variance', params_shape, initializer=tf.ones_initializer, trainable=False)
    #train_ops.append(moving_averages.assign_moving_average( moving_mean, mean, 0.9))
    #train_ops.append(moving_averages.assign_moving_average( moving_variance, variance, 0.9))

    #mean, variance = control_flow_ops.cond( is_training, lambda: (mean, variance), lambda: (moving_mean, moving_variance))

    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)


def conv(x,ksize, output_channel, stride):
    print "conv x:",x.shape, ksize, output_channel
    xshape = [ksize, ksize, x.get_shape()[-1], output_channel]
    #initializer = tf.truncated_normal_initializer(mean=0.0,stddev=0.1)
    initializer = tf.contrib.layers.xavier_initializer( uniform=False )

    with tf.name_scope('conn_weights'):
        weights = get_variable('weights', shape=xshape, dtype='float', initializer=initializer)
        variable_summaries(weights)

    x=tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')
    print "x:",x.shape, ksize, output_channel
    return x

def relu(x):
    x=bn(x)
    return tf.nn.relu(x)

def conv_block(x, output_channels, begin=0):
    if begin == 0 :
        stride=2
    else:
        stride=1
    print "block ",begin, x.shape
    short_cut = x

    with tf.variable_scope('block1'):
        x=conv(x,3, output_channels,stride)
        x=relu(x)

    with tf.variable_scope('block2'):
        x=conv(x, 3, output_channels,1)

    with tf.variable_scope('short_cut'):
        if x.shape[-1] != short_cut.shape[-1]:
            short_cut=conv(short_cut,1, x.shape[-1],2)
            print "short cut conv:",short_cut.shape
        x+=short_cut
        x=relu(x)
    print "block done:",begin, x.shape
    return x
   
def max_pool(x, ksize, stride):
    return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding='SAME')

def avg_pool(x):
    return tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")

def fc(x, out_d):
    print "fc:",x.shape
    num_units_in = x.get_shape()[1]
    num_units_out =  out_d
    weights_initializer = tf.truncated_normal_initializer(stddev=0.01)

    weights = get_variable('weights', shape=[num_units_in, num_units_out], initializer=weights_initializer)
    biases = get_variable('biases', shape=[num_units_out], initializer=tf.zeros_initializer)
    x = tf.nn.xw_plus_b(x, weights, biases)
    print "fc:",x
    return x

def pre_process_image(image):
# Randomly crop the input image.
    img_size_cropped = 32
    num_channels = 3

    image = tf.pad(image, [[4,4],[4,4],[0,0]], "CONSTANT")
    print "image;",image.shape

    image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)
    
    # Randomly adjust hue, contrast and saturation.
    #image = tf.image.random_hue(image, max_delta=0.05)
    #image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
    #image = tf.image.random_brightness(image, max_delta=0.2)
    #image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

    # Some of these functions may overflow and result in pixel
    # values beyond the [0, 1] range. It is unclear from the
    # documentation of TensorFlow 0.10.0rc0 whether this is
    # intended. A simple solution is to limit the range.

    # Limit the image pixels between [0, 1] in case of overflow.
    #image = tf.minimum(image, 1.0)
    #image = tf.maximum(image, 0.0)
    return image

def pre_large_process_image(image):
# Randomly crop the input image.
    img_size_cropped = 224
    num_channels = 3

    print "image;",image.shape

    image = tf.resize_image_with_crop_or_pad(image, [img_size_cropped, img_size_cropped])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)
    
    # Randomly adjust hue, contrast and saturation.
    #image = tf.image.random_hue(image, max_delta=0.05)
    #image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
    #image = tf.image.random_brightness(image, max_delta=0.2)
    #image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

    # Some of these functions may overflow and result in pixel
    # values beyond the [0, 1] range. It is unclear from the
    # documentation of TensorFlow 0.10.0rc0 whether this is
    # intended. A simple solution is to limit the range.

    # Limit the image pixels between [0, 1] in case of overflow.
    #image = tf.minimum(image, 1.0)
    #image = tf.maximum(image, 0.0)
    return image


def pre_process(images):
    return tf.map_fn(lambda image: pre_process_image(image), images)

def pre_large_process(images):
    return tf.map_fn(lambda image: pre_large_process(image), images)

def get_decay():
    costs = []
    for var in tf.trainable_variables():
        if var.op.name.find(r'weights') > 0:
            costs.append(tf.nn.l2_loss(var))
    print costs
    return tf.multiply(weight_decay_rate, tf.add_n(costs))

def get_loss(x,y):
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x, labels=y))
    loss = cross_entropy + get_decay()
    return loss

class SmallResNet:
    def __init__(self):
        self.num_class = 10
        self.x = tf.placeholder(tf.float32, [None, 32,32,3])
        self.y = tf.placeholder(tf.int32, [None])

        self.logit=self.build_model(self.x)
        self.loss=get_loss(self.logit,self.y)
        print "logit:",self.logit.shape
        self.predict = tf.nn.softmax(self.logit)

        with tf.variable_scope('loss'):
            variable_summaries(self.loss)

    def build_model(self,x):
        with tf.variable_scope('input'):
            variable_summaries(x)

        with tf.variable_scope('conv0'):
            x=conv(x,3,16,1)
            with tf.variable_scope('input'):
                variable_summaries(x)
            x=relu(x)

        stack_num = 5
        for i in xrange(1,stack_num+1):
            with tf.variable_scope('conv1_'+str(i-1)):
                x=conv_block(x,16,1)
                with tf.variable_scope('output'):
                    variable_summaries(x)

        for i in xrange(0,stack_num):
            with tf.variable_scope('conv2_'+str(i)):
                x=conv_block(x,32,i)
                with tf.variable_scope('input'):
                    variable_summaries(x)

        for i in xrange(0,stack_num):
            with tf.variable_scope('conv3_'+str(i)):
                x=conv_block(x,64,i)
                with tf.variable_scope('output'):
                    variable_summaries(x)
                    
        with tf.variable_scope('out'):
            x=avg_pool(x)
            x=fc(x,self.num_class)
            with tf.variable_scope('output'):
                variable_summaries(x)
        print x.shape

        return x

class ResNet:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 224,224,3])
        self.y = tf.placeholder(tf.int32, [None])
        num_class=1000

        self.logit=self.build_model(self.x,num_class)
        self.loss=get_loss(self.logit,self.y)
        print "logit:",self.logit.shape
        self.predict = tf.nn.softmax(self.logit)

        with tf.variable_scope('loss'):
            variable_summaries(self.loss)

    def build_model(self,x, num_class):
        with tf.variable_scope('input'):
            variable_summaries(x)

        with tf.variable_scope('conv0'):
            x=conv(x,7,64,2)#7*7, 64, stride=2
            x=relu(x)
            x=max_pool(x, 3, 2)#3*3 stride=2, max pool
            with tf.variable_scope('input'):
                variable_summaries(x)
        print "x input:",x.shape
        stack_num=[3,4,6,3]
        for i in xrange(1,stack_num[0]+1):
            with tf.variable_scope('conv1_'+str(i-1)):
                x=conv_block(x,64,1)
                with tf.variable_scope('output'):
                    variable_summaries(x)

        for i in xrange(0,stack_num[1]):
            with tf.variable_scope('conv2_'+str(i)):
                x=conv_block(x,128,i)
                with tf.variable_scope('input'):
                    variable_summaries(x)

        for i in xrange(0,stack_num[2]):
            with tf.variable_scope('conv3_'+str(i)):
                x=conv_block(x,256,i)
                with tf.variable_scope('output'):
                    variable_summaries(x)

        for i in xrange(0,stack_num[3]):
            with tf.variable_scope('conv4_'+str(i)):
                x=conv_block(x,512,i)
                with tf.variable_scope('output'):
                    variable_summaries(x)
                    
        with tf.variable_scope('out'):
            x=avg_pool(x)
            x=fc(x,num_class)
            with tf.variable_scope('output'):
                variable_summaries(x)
        print x.shape

        return x

#r = ResNet()
#print r.loss
