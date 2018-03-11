import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
from component import component
import datetime

class ResNet:
    def __init__(self, images, labels, num_class, mode, stack_num=[5,5,5], conv_layers=[16,32,64]):
        self.num_class = num_class
        
        self.x = images
        self.y = labels
        self.mode = mode

        self.stack_num = stack_num
        self.conv_layers = conv_layers
        self.train_ops = []
        self.build_model(self.x)

    def build_model(self,x):
        with tf.variable_scope('input'):
            component.variable_summaries(x)

        with tf.variable_scope('conv0'):
            x=component.conv(x,3,self.conv_layers[0],1)
            with tf.variable_scope('input'):
                component.variable_summaries(x)
            x=component.bn(x,self.mode, self.train_ops)
            x=component.relu(x)

        for i in xrange(1,self.stack_num[0]+1):
            with tf.variable_scope('conv1_'+str(i-1)):
                x=self.conv_block(x,self.conv_layers[0],i)
                with tf.variable_scope('output'):
                    component.variable_summaries(x)

        for i in xrange(0,self.stack_num[1]):
            with tf.variable_scope('conv2_'+str(i)):
                x=self.conv_block(x,self.conv_layers[1],i)
                with tf.variable_scope('input'):
                    component.variable_summaries(x)

        for i in xrange(0,self.stack_num[2]):
            with tf.variable_scope('conv3_'+str(i)):
                x=self.conv_block(x,self.conv_layers[2],i)
                with tf.variable_scope('output'):
                    component.variable_summaries(x)
                    
        with tf.variable_scope('out'):
            x=component.avg_pool(x)
            x=component.fc(x,self.num_class)
            with tf.variable_scope('output'):
                component.variable_summaries(x)
        print x.shape

        self.logit=x
        self.loss=component.get_softmax_loss(self.logit,self.y)
        self.predict = tf.nn.softmax(self.logit)
        tmp_pred = tf.to_int32(tf.argmax(self.predict, axis=1))
        self.precision = tf.reduce_mean(tf.to_float(tf.equal(tmp_pred, self.y)))

        with tf.variable_scope('loss'):
            component.variable_summaries(self.loss)
        return x

    def conv_block(self,x, output_channels, begin=0):
        if begin == 0 :
            stride=2
        else:
            stride=1
        print "block ",begin, x.shape
        short_cut = x

        with tf.variable_scope('block1'):
            x=component.conv(x,3, output_channels,stride)
            x=component.bn(x,self.mode, self.train_ops)
            x=component.relu(x)

        with tf.variable_scope('block2'):
            x=component.conv(x, 3, output_channels,1)

        with tf.variable_scope('short_cut'):
            if x.shape[-1] != short_cut.shape[-1]:
                short_cut=component.conv(short_cut,1, x.shape[-1],2)
                print "short cut conv:",short_cut.shape
            x+=short_cut
            x=component.bn(x,self.mode, self.train_ops)
            x=component.relu(x)
        print "block done:",begin, x.shape
        return x

    def get_train_op(self,global_step, lr_rate, moment_rate=0.9):

        optimizer = tf.train.MomentumOptimizer(lr_rate,moment_rate)
        grads = optimizer.compute_gradients(self.loss)

        for grad, var in grads:
            if grad is not None :
                tf.summary.histogram(var.op.name + '/gradients', grad)
                print grad
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

        train_ops = [apply_gradient_op] + self.train_ops
        return tf.group(*train_ops)
