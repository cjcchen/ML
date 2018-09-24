from vgg19 import Vgg19
import tensorflow as tf
import numpy as np
import math

import data
import utils

import os

tf.flags.DEFINE_string("caption_path", None,
                    "Where the training/test data is stored.")
tf.flags.DEFINE_string("image_path", None,
                    "Model output directory.")

tf.flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
tf.flags.DEFINE_string("save_path", None,
                    "Model output directory.")
tf.flags.DEFINE_string("log_path", None,
                    "Model output directory.")

FLAGS = tf.flags.FLAGS

class Image_Attention:
    def __init__(self, w2d, config):
        self.config = config
        word_to_idx=w2d
        self.w2d = word_to_idx
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in w2d.iteritems()}

        self.target = tf.placeholder(tf.int32, [None,None])
        self.word = tf.placeholder(tf.int32, [None,None])
        self.feature = tf.placeholder(tf.float32, [None, 196,512])

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        self.config.vob_size = len(w2d)
        self._null = w2d['<NULL>']
        self._start = self.w2d['<START>']

    def build(self):
        image_feature = self.bn(self.feature)
        inputs = self.embedding (self.word)

        self.init_c,self.init_h = self.get_init_state(image_feature)
        c = self.init_c
        h = self.init_h

        lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=self.config.hidden_size, reuse=tf.AUTO_REUSE)
        #lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.config.hidden_size)

        alpha_list = []
        loss = 0.0
        x=inputs
        for time_step in range(self.config.seq_len):
            context, alpha = self.attention_layer(image_feature, h, reuse=(time_step!=0))
            alpha_list.append(alpha)

            context, _ = self.selector(context, h, reuse=(time_step!=0))

            with tf.variable_scope('lstm', reuse=(time_step!=0)):
                _, (c, h) = lstm_cell(inputs=tf.concat( [x[:,time_step,:], context],1), state=[c, h])

            y = self.decode_layer(context, h, inputs[:,time_step,:], dropout=True, reuse=(time_step!=0))

            loss += tf.reduce_sum( tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target[:, time_step],logits=y))

        alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))     # (N, T, L)
        alphas_all = tf.reduce_sum(alphas, 1)      # (N, L)
        alpha_reg = 1.0 * tf.reduce_sum((16./196 - alphas_all) ** 2)
        loss += alpha_reg

        return loss/tf.to_float(tf.shape(inputs)[0])


    def build_sampler(self, max_len):
        image_feature = self.bn(self.feature, mode='test')
        self.init_c,self.init_h = self.get_init_state(image_feature)
        c = self.init_c
        h = self.init_h

        lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=self.config.hidden_size, reuse=tf.AUTO_REUSE)
        #lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.config.hidden_size)

        start = tf.fill([tf.shape(image_feature)[0]], self.w2d['<START>'])
        current_word=start
        alpha_list = []
        beta_list = []
        sampled_word_list = []

        for time_step in range(max_len):
            x = self.embedding (current_word,reuse=(time_step!=0))
            context, alpha = self.attention_layer(image_feature, h,reuse=(time_step!=0))
            alpha_list.append(alpha)

            context, beta = self.selector(context, h,reuse=(time_step!=0))
            beta_list.append(beta)
            with tf.variable_scope('lstm', reuse=(time_step!=0)):
                _, (c, h) = lstm_cell(inputs=tf.concat([x,context], 1), state=[c, h])

            logits = self.decode_layer(context, h, x,reuse=(time_step!=0))
            current_word = tf.argmax(logits, 1)
            sampled_word_list.append(current_word)

        alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))     # (N, T, L)
        betas = tf.transpose(tf.squeeze(beta_list), (1, 0))    # (N, T)
        sampled_captions = tf.transpose(tf.stack(sampled_word_list), (1, 0))     # (N, max_len)
        return alphas,betas, sampled_captions



    def init_state(self, sess, image):
        return sess.run( [self.init_c, self.init_h], feed_dict={ self.image: image })


    def predict_one(self, sess, image, word,c,h):
        return sess.run([self.logits, self.next_c, self.next_h], feed_dict= {
            self.image:image,
            self.word:word,
            self.c:c,
            self.h:h
            })

    def decode_layer(self, context, h, input, dropout = False, reuse=False):
        with tf.variable_scope('decode_layer', reuse=reuse):
            l_dim, d_dim = self.config.feature_dim
            assert len(context.get_shape())==2
            assert context.get_shape()[1] == d_dim
            assert len(h.get_shape())==2
            assert h.get_shape()[1] == self.config.hidden_size
            assert len(input.get_shape())==2
            assert input.get_shape()[1] == self.config.embedding_size

            #do output embedding first, with tanh and bias
            with tf.variable_scope('output_h'):
                h_w = tf.get_variable('output_w', [self.config.hidden_size, self.config.embedding_size], initializer=self.weight_initializer)
                h_b = tf.get_variable('output_b', [self.config.embedding_size], initializer=self.const_initializer)

                if dropout:
                    h = tf.nn.dropout(h, 0.5)
                h = tf.matmul(h, h_w) + h_b

            with tf.variable_scope('output_z'):
                z_w = tf.get_variable('output_w', [d_dim, self.config.embedding_size], initializer=self.weight_initializer)

                z = tf.matmul(context, z_w)

            with tf.variable_scope('output_y'):
                y = h+z+input
                y = tf.nn.tanh(y)

            with tf.variable_scope('output'):
                if dropout:
                    y = tf.nn.dropout(y, 0.5)
                y_w = tf.get_variable('output_w', [self.config.embedding_size, self.config.vob_size], initializer=self.weight_initializer)
                y_b = tf.get_variable('output_b', [self.config.vob_size], initializer=self.const_initializer)
                y = tf.matmul(y,y_w) + y_b

        return y

    def attention_layer(self, image_feature, h, reuse=False):
        with tf.variable_scope('attention_layer', reuse=reuse):
            l_dim, d_dim = self.config.feature_dim

            i_w = tf.get_variable('image_w', [d_dim, d_dim], initializer=self.weight_initializer)
            print "get var:",i_w
            h_w = tf.get_variable('h_w', [self.config.hidden_size, d_dim], initializer=self.weight_initializer)
            b = tf.get_variable('h_b', [d_dim], initializer=self.const_initializer)

            image_feature_flat = tf.reshape(image_feature, [-1, d_dim])
            image_feature_proj = tf.matmul(image_feature_flat, i_w)
            image_feature_proj = tf.reshape(image_feature_proj, [-1, l_dim, d_dim])

            h_att = tf.matmul(h, h_w)

            x = tf.expand_dims(h_att, 1) + image_feature_proj + b
            x = tf.nn.relu(x) #[-1,L,D]
            x = tf.reshape(x,[-1,d_dim]) #flat for mul

            o_w = tf.get_variable('o_w', [d_dim, 1], initializer=self.weight_initializer)

            out = tf.matmul(x, o_w)
            out = tf.reshape(out, [-1,l_dim]) #[-1,L]

            alpha = tf.nn.softmax(out) #[-1,L]

            print ("alpha shape:",alpha.shape)

            context = tf.reduce_sum(image_feature * tf.expand_dims(alpha, 2),1) #cal each L's weight,

            return context, alpha

    def assign_lr(self, sess, lr):
        sess.run(self.lr_op, feed_dict={self.new_lr:lr})

    def get_init_state(self, image_feature):
        print "im shape:",image_feature.shape
        image_feature_mean = tf.reduce_mean(image_feature, 1)
        print "mean shape:",image_feature_mean.shape

        with tf.variable_scope('init_c'):
            c = self.fc(image_feature_mean, self.config.hidden_size, bias=True, func=tf.nn.tanh)

        with tf.variable_scope('init_h'):
            h = self.fc(image_feature_mean, self.config.hidden_size, bias=True, func=tf.nn.tanh)

        return c,h

    def embedding (self, inputs, reuse=False):
        print "input shape:",inputs.shape
        print type(inputs)
        with tf.variable_scope('word_embedding', reuse=reuse):
            w = tf.get_variable('embedding_w', [self.config.vob_size, self.config.embedding_size], dtype=tf.float32, initializer=self.emb_initializer)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return x

    def selector(self, context, h, reuse=False):
        with tf.variable_scope('selector', reuse=reuse):
            w = tf.get_variable('selector_w', [self.config.hidden_size, 1], initializer=self.weight_initializer)
            b = tf.get_variable('selector_b', [1], initializer=self.const_initializer)
            beta = tf.nn.sigmoid(tf.matmul(h, w) + b, 'beta')    # (N, 1)
            print "beta shape:",beta.shape, "context shape:",context.shape
            context = tf.multiply(beta, context, name='selected_context')
            print "after:",context.shape
            return context, beta

    def fc(self, input, output_dim, bias=True, func=None):
        with tf.variable_scope('fc'):
            w = tf.get_variable('fc_w', [input.get_shape()[-1], output_dim], initializer=self.weight_initializer)
            o = tf.matmul(input, w)

            if bias:
                b = tf.get_variable('fc_b', [output_dim], initializer=self.const_initializer)
                o = o + b

            if func:
                o = func(o)

            return o

    def bn(self,x, mode='train'):
               return tf.contrib.layers.batch_norm(inputs=x,
                       decay=0.95,
                       center=True,
                       scale=True,
                       is_training=(mode=='train'),
                       updates_collections=None,
                       scope=('batch_norm'))

    '''
    def run_epoch(self,sess, lr_decay, epoch_size, summary_writer,sv):
        summary = tf.Summary()
        epochs = 0
        costs = 0
        for e in range(epoch_size):
            feed_dict={}
            cost,_= sess.run([self._cost, self._train_op])
            costs += cost
            epochs += self.config.seq_len
            if e % 10 == 0:
                perxity=np.exp(costs / epochs)
                summary.value.add(tag='perxity', simple_value=perxity)
                summary.value.add(tag='lr', simple_value=lr_decay)
                i_global=sess.run(tf.train.get_or_create_global_step())
                print ("pt %f cost %f pre cost %f epo %d global step %d" % (perxity, costs, cost, epochs, i_global))
                summary_writer.add_summary(summary, i_global)#write eval to tensorboard
                #summary_writer.add_summary(summary_str, i_global)#write eval to tensorboard
            if e % 100 == 0:
                save_model(sess, sv, FLAGS.save_path, i_global)
        return perxity
    '''

    def run_epoch(self,sess, lr_decay, epoch_size, summary_writer,sv):
        cost,_,lr,sum_str= sess.run([self._cost, self._train_op, self.lr, self.summary_op])
        return cost,lr,sum_str


if __name__ == '__main__':
    train()
    #predict()

