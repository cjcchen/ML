
import tensorflow as tf
from component import component
from component import network_base


class ZFNet(network_base.BaseNet):
    def __init__(self, images, labels, num_class, mode, pre_fix):
        network_base.BaseNet.__init__(self,pre_fix)

        self.x = images
        self.y = labels
        self.num_class = num_class
        self.mode = mode
        self.train_ops = []

        self.conv_layer=[]
        self.build_model()

    def build_model(self):
        x = self.x
        with tf.variable_scope('conv1'):
            conv0_x = component.conv(x,7,96,2)
            conv0_x = component.bn(conv0_x,self.mode, self.train_ops)
            conv0_x = component.relu(conv0_x)
            conv1_x = component.max_pool(conv0_x,3,stride=2)
            self.conv_layer.append(conv1_x)

        with tf.variable_scope('conv2'):
            conv2_x = component.conv(conv1_x,ksize=5,output_channel=256,stride=2)
            conv2_x = component.bn(conv2_x,self.mode, self.train_ops)
            conv2_x = component.relu(conv2_x)
            conv2_x = component.max_pool(conv2_x,3,stride=2)
            self.conv_layer.append(conv2_x)

        with tf.variable_scope('conv3'):
            component.variable_summaries(conv2_x)
            conv3_x = component.conv(conv2_x,ksize=3,output_channel=384,stride=1)
            conv3_x = component.bn(conv3_x,self.mode, self.train_ops)
            conv3_x = component.relu(conv3_x)
            self.conv_layer.append(conv3_x)

        with tf.variable_scope('conv4'):
            component.variable_summaries(conv3_x)
            conv4_x = component.conv(conv3_x,ksize=3,output_channel=384,stride=1)
            conv4_x = component.bn(conv4_x,self.mode, self.train_ops)
            conv4_x = component.relu(conv4_x)
            self.conv_layer.append(conv4_x)

        with tf.variable_scope('conv5'):
            component.variable_summaries(conv4_x)
            conv5_x = component.conv(conv4_x,ksize=3,output_channel=256,stride=1)
            conv5_x = component.bn(conv5_x,self.mode, self.train_ops)
            conv5_x = component.relu(conv5_x)
            conv5_x = component.max_pool(conv5_x,3,stride=2)
            self.conv_layer.append(conv5_x)

        if self.mode != "tunning":
            p = conv5_x.get_shape().as_list()
            flat_shape = p[1]*p[2]*p[3]
            with tf.variable_scope('fc1'):
                fc1_x = tf.reshape(conv5_x, [-1, flat_shape])
                #fc1_x = component.bn(fc1_x,self.mode, self.train_ops)
                component.variable_summaries(fc1_x)
                fc1_x = component.relu(fc1_x)

            with tf.variable_scope('fc2'):
                fc2_x = component.fc(fc1_x,4096)
                #fc2_x = component.bn(fc2_x,self.mode, self.train_ops)
                component.variable_summaries(fc2_x)
                fc2_x = component.relu(fc2_x)

            with tf.variable_scope('fc3'):
                fc3_x = component.fc(fc2_x,self.num_class)
                #fc3_x = component.bn(fc3_x,self.mode, self.train_ops)
                component.variable_summaries(fc3_x)
                fc3_x = component.relu(fc3_x)


            self.logit = fc3_x
            self.predict = tf.nn.softmax(self.logit)
            tmp_pred = tf.to_int32(tf.argmax(self.predict, axis=1))

            if(self.mode == 'train'):
                self.loss=component.get_softmax_loss(self.logit,self.y)
                self.precision = tf.reduce_mean(tf.to_float(tf.equal(tmp_pred, self.y)))

    def get_conv_layer(self,index):
        return self.conv_layer[index]

if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [None,224,224,3])
    y = tf.placeholder(tf.int32, [None] )

    model = ZFNet(x,y,1000,'train')
