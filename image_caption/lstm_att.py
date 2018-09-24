from vgg19 import Vgg19
import tensorflow as tf
import numpy as np
import math

import data
import utils

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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

class IMAGE_ATT_CAP:
    def __init__(self, image, word, target, config, is_training=True):
        self.image = image
        self.word = word
        self.target = target
        self.config = config
        self.summaries = []

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        self.L = self.config.feature_dim[0]
        self.D = self.config.feature_dim[1]
        self.H = self.config.hidden_size
        self.build(is_training)

    def _batch_norm(self, x, mode='train', name=None):
        return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=(mode=='train'),
                                            updates_collections=None,
                                            scope=(name+'batch_norm'))

    def _attention_layer(self, features, features_proj, h, reuse=False):
        with tf.variable_scope('attention_layer', reuse=reuse):
            w = tf.get_variable('w', [self.H, self.D], initializer=self.weight_initializer)
            b = tf.get_variable('b', [self.D], initializer=self.const_initializer)
            w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)

            h_att = tf.nn.relu(features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b)    # (N, L, D)
            out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.L])   # (N, L)
            alpha = tf.nn.softmax(out_att)
            context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')   #(N, D)
            return context, alpha

    def _project_features(self, features):
        with tf.variable_scope('project_features'):
            w = tf.get_variable('w', [512, 512], initializer=self.weight_initializer)
            features_flat = tf.reshape(features, [-1, self.D])
            features_proj = tf.matmul(features_flat, w)
            features_proj = tf.reshape(features_proj, [-1, self.L, self.D])
            return features_proj

    def build(self,is_training):
            with tf.name_scope("content_vgg"):
                self.cnn_net = Vgg19()
                self.cnn_net.build(self.image)

            image_feature = self.cnn_net.conv5_3 #[-1,14*14,512]
            image_feature = tf.reshape(image_feature, [-1]+ self.config.feature_dim)
            print ("image feature shape:",image_feature)

            image_feature = self.bn(image_feature)

            print ("image feature shape:",image_feature)

            #features_proj = self._project_features(features=image_feature)

            inputs = self.embedding (self.word)
            self.init_c,self.init_h = self.get_init_state(image_feature)
            if is_training:
                c = self.init_c
                h = self.init_h
            else:
                self.c = tf.placeholder(tf.float32, [None, self.config.hidden_size])
                self.h = tf.placeholder(tf.float32, [None, self.config.hidden_size])
                c = self.c
                h = self.h


            lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=self.config.hidden_size, reuse=tf.AUTO_REUSE)
            if is_training and self.config.keep_prob < 1:
                lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.config.keep_prob)

            alpha_list = []
            loss = 0.0
            with tf.variable_scope('layer'):
                for time_step in range(self.config.seq_len):
                    if time_step > 0: tf.get_variable_scope().reuse_variables()
                    context, alpha = self.attention_layer(image_feature, h)
                    #context, alpha = self._attention_layer(image_feature, features_proj, h, reuse=(time_step!=0))
                    alpha_list.append(alpha)

                    context, _ = self.selector(context, h)

                    x = tf.concat([inputs[:,time_step,:], context],1)
                    _, (c, h) = lstm_cell(inputs=x, state=[c, h])

                    y = self.decode_layer(context, h, inputs[:,time_step,:])

                    if is_training:
                        loss += tf.reduce_sum( tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target[:, time_step],logits=y))

            self.logits = tf.nn.softmax(y)

            self.next_c = c
            self.next_h = h

            if is_training:
                alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))     # (N, T, L)
                alphas_all = tf.reduce_sum(alphas, 1)      # (N, L)
                alpha_reg = 1.0 * tf.reduce_sum((1 - alphas_all) ** 2)
                #alpha_reg = 1.0 * tf.reduce_sum((16./196 - alphas_all) ** 2)
                #loss += alpha_reg

                self._cost = loss/tf.to_float(tf.shape(inputs)[0])

                tvars = tf.trainable_variables()
                grads = tf.gradients(self._cost, tvars)
                grads_and_vars = zip(grads, tf.trainable_variables())
                #grads, _= tf.clip_by_global_norm(tf.gradients(self._cost, tvars), self.config.max_grad_norm)

                self.lr = tf.Variable(0.0, trainable=False)
                self.new_lr = tf.Variable(0.0, trainable=False)
                self.lr_op=tf.assign(self.lr,self.new_lr)

                #grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars), config.max_grad_norm)
                optimizer = tf.train.GradientDescentOptimizer(self.lr)
                self._train_op = optimizer.apply_gradients(
                    grads_and_vars,
                    global_step=tf.train.get_or_create_global_step())


                # summary op
                # tf.scalar_summary('batch_loss', loss)
                for var in tf.trainable_variables():
                    #tf.histogram_summary(var.op.name, var)
                    tf.summary.histogram(var.op.name, var)
                for grad, var in grads_and_vars:
                    #tf.histogram_summary(var.op.name+'/gradient', grad)
                    tf.summary.histogram(var.op.name+'/gradient', grad)

                self.summary_op = tf.summary.merge_all()


    def init_state(self, sess, image):
        return sess.run( [self.init_c, self.init_h], feed_dict={ self.image: image })


    def predict_one(self, sess, image, word,c,h):
        return sess.run([self.logits, self.next_c, self.next_h], feed_dict= {
            self.image:image,
            self.word:word,
            self.c:c,
            self.h:h
            })

    def decode_layer(self, context, h, input):
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

            h = tf.nn.dropout(h, 0.5)
            h = tf.matmul(h, h_w) + h_b

        with tf.variable_scope('output_z'):
            z_w = tf.get_variable('output_w', [d_dim, self.config.embedding_size], initializer=self.weight_initializer)

            z = tf.matmul(context, z_w)

        with tf.variable_scope('output_y'):
            y = h+z+input
            y = tf.nn.tanh(y)

        with tf.variable_scope('output'):
            y = tf.nn.dropout(y, 0.5)
            y_w = tf.get_variable('output_w', [self.config.embedding_size, self.config.vob_size], initializer=self.weight_initializer)
            y_b = tf.get_variable('output_b', [self.config.vob_size], initializer=self.const_initializer)
            y = tf.matmul(y,y_w) + y_b

        return y

    def attention_layer(self, image_feature, h):
        l_dim, d_dim = self.config.feature_dim

        i_w = tf.get_variable('image_w', [d_dim, d_dim], initializer=self.weight_initializer)
        h_w = tf.get_variable('h_w', [self.config.hidden_size, d_dim], initializer=self.weight_initializer)
        b = tf.get_variable('b', [d_dim], initializer=self.const_initializer)

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
        print "if shape:",image_feature.shape
        image_feature_mean = tf.reduce_mean(image_feature, 1)
        print "mean shape:",image_feature_mean.shape

        with tf.variable_scope('init_c'):
            c = self.fc(image_feature_mean, self.config.hidden_size, bias=True, func=tf.nn.tanh)

        with tf.variable_scope('init_h'):
            h = self.fc(image_feature_mean, self.config.hidden_size, bias=True, func=tf.nn.tanh)

        return c,h

    def embedding (self, inputs):
        print "input shape:",inputs.shape
        with tf.variable_scope('word_embedding'):
            w = tf.get_variable('w', [self.config.vob_size, self.config.embedding_size], dtype=tf.float32, initializer=self.emb_initializer)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return x

    def selector(self, context, h):
        with tf.variable_scope('selector'):
            w = tf.get_variable('w', [self.config.hidden_size, 1], initializer=self.weight_initializer)
            b = tf.get_variable('b', [1], initializer=self.const_initializer)
            beta = tf.nn.sigmoid(tf.matmul(h, w) + b, 'beta')    # (N, 1)
            print "beta shape:",beta.shape, "context shape:",context.shape
            context = tf.multiply(beta, context, name='selected_context')
            print "after:",context.shape
            return context, beta

    def fc(self, input, output_dim, bias=True, func=None):
        with tf.variable_scope('fc'):
            w = tf.get_variable('w', [input.get_shape()[-1], output_dim], initializer=self.weight_initializer)
            o = tf.matmul(input, w)

            if bias:
                b = tf.get_variable('b', [output_dim], initializer=self.const_initializer)
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


def save_model(sess, sv, save_path, global_step):
    sv.save(sess, save_path=save_path, global_step=global_step)


class img_config():
    hidden_size=1024
    vob_size=20000
    embedding_size=512
    feature_dim = [196,512]
    seq_len = 20
    num_layers = 2

    init_scale = 0.1
    learning_rate = 0.001
    max_grad_norm = 5
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20


class Sentence:
    def __init__(self, start_w, score,c,h):
        self.score = score
        self.words = start_w
        self.c = c
        self.h = h

def load_session(sess, checkpoint_dir):
    saver = tf.train.Saver()

    try:
        print("Trying to restore last checkpoint ...:",checkpoint_dir)
        last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
        saver.restore(sess, save_path=last_chk_path)
        print("restore last checkpoint %s done"%checkpoint_dir)
    except Exception as e:
        print("Failed to restore checkpoint. Initializing variables instead."),e
        sess.run(tf.global_variables_initializer())
        sess.run(tf.initialize_all_variables())

    return saver


def train():
    config = img_config()
    config.batch_size=128
    f, image,label,word, target, w2d,d2w = data.get_data(FLAGS.caption_path, FLAGS.image_path, max_len=config.seq_len+1, batch_size=config.batch_size)
    epoch_size=10000
    config.vob_size = len(w2d)
    print ("vb size:",len(w2d))
    image_caption = IMAGE_ATT_CAP(image, word, target, config)


#summary_op = tf.merge_all_summaries()

    #sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    config_proto = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
    #with sv.managed_session(config=config_proto) as sess:
    with tf.Session(config=config_proto) as sess:
        sv = load_session(sess, FLAGS.save_path)
        threads = tf.train.start_queue_runners(sess)

        summary_writer = tf.summary.FileWriter(FLAGS.log_path,sess.graph)

        for i in range(config.max_max_epoch):
            x_lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
            print ("lr:",x_lr_decay)
            image_caption.assign_lr(sess, config.learning_rate * x_lr_decay)

            for j in range(10000):
                loss,lr,sum_str=image_caption.run_epoch(sess, x_lr_decay, epoch_size, summary_writer,sv)
                if j % 10==0:
                    print ("step %d per %f, lr %f" % (i,loss,lr))
                    summary = tf.Summary()
                    summary.value.add(tag='loss', simple_value=loss)
                    i_global=sess.run(tf.train.get_or_create_global_step())
                    print ("cost %f global step %d" % (loss, i_global))
                    summary_writer.add_summary(summary, i_global)#write eval to tensorboard
                    summary_writer.add_summary(sum_str, i_global)

                    if j % 100 == 0:
                        save_model(sess, sv, FLAGS.save_path, i_global)


def predict():
    config = img_config()
    config.seq_len=1
    #img = utils.load_image("/home/tusimple/junechen/ml_data/data/train2014/COCO_train2014_000000160629.jpg")
    img = utils.load_image("/home/tusimple/junechen/ml_data/data/train2014/COCO_train2014_000000318556.jpg")
    #img = utils.load_image("./test_data/tiger.jpeg")
    img = img.reshape((1, 224, 224, 3))
    w2d, d2w = data.get_word_to_id()
    config.vob_size = len(w2d)
    print "read w2d size:",len(w2d)
    if len(w2d) == 0:
        f, image,label, word, target, w2d, d2w = data.get_data(FLAGS.caption_path, FLAGS.image_path, max_len=16, batch_size=config.batch_size)

    images = tf.placeholder("float", [None, 224, 224, 3], name="image")
    word = tf.placeholder(tf.int32, [None, None], name="word_seq")

    image_caption = IMAGE_ATT_CAP(images, word, None, config, is_training=False)
    config_proto = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config_proto) as sess:
        sv = load_session(sess, FLAGS.save_path)
        c, h = image_caption.init_state(sess, img)

        candidats = [Sentence([w2d['<start>']],0, c,h)]

        #print c, h
        for i in range(10):
            new_list = []
            print ("cant len:",len(candidats))
            seq = np.array([ s.words[-1] for s in candidats ]).reshape(len(candidats),-1)
            c_list = np.vstack([ s.c for s in candidats ]).reshape(len(candidats),-1)
            h_list = np.vstack([ s.h for s in candidats ]).reshape(len(candidats),-1)


            images=np.array([img]*len(seq)).reshape(-1,224,224,3)

            #print "seq shape:",seq.shape
            #print "c list shape:",c_list.shape
            #print "h list shape:",h_list.shape
            #print "seq:",seq
            #if len(c_list) > 1:
            #    print "c list:",c_list[1]
            #    print "h list:",h_list[1]

            #outputs, c, h = image_caption.predict_one(sess, images, seq,c_list,h_list)

            output,n_c,n_h = sess.run([image_caption.logits,image_caption.next_c,image_caption.next_h], feed_dict={
                image_caption.image:images,
                image_caption.word:seq,
                image_caption.c:c_list,
                image_caption.h:h_list
                })
            #print output.shape
            #print output
            #print n_c.shape
            #print n_h.shape
            #print c
            #print h
            #predict = output[0][-1]
            #sort_idx = predict.argsort()[::-1]
            #print sort_idx

            new_list = []
            for s, s_out,s_c,s_h in zip(candidats, output,n_c,n_h):
                predict = s_out
                sort_idx = predict.argsort()[::-1]
                #print "old:",s.words
                #print "out:",s_out
                #print "index:",sort_idx
                #print "c:",s_c
                #print "h:",s_h
                for j in range(len(sort_idx)):
                    #print "S:",s.words,s.words+[sort_idx[j]]
                    w = sort_idx[j]
                    score = np.log(predict[w])
                    if math.isnan(score):
                        continue
                    new_list.append(Sentence(s.words+[w], s.score + score,s_c,s_h))

            new_list=sorted(new_list, key=lambda sentence: sentence.score, reverse=True)
            candidats = new_list[0:20]
            for l in candidats:
                print ("sort list:",l.words, l.score)
                print [ d2w[p] for p in l.words]

def predict1():
    config = img_config()
    config.seq_len=2
    img = utils.load_image("/home/tusimple/junechen/ml_data/data/train2014/COCO_train2014_000000318556.jpg")
    #img = utils.load_image("/home/tusimple/junechen/ml_data/data/train2014/COCO_train2014_000000160629.jpg")
    #/home/tusimple/junechen/ml_data/data/train2014/COCO_train2014_000000318556.jpg
    #img = utils.load_image("./test_data/tiger.jpeg")
    img = img.reshape((1, 224, 224, 3))
    w2d, d2w = data.get_word_to_id()
    config.vob_size = len(w2d)
    print "read w2d size:",len(w2d)
    if len(w2d) == 0:
        f, image,label, word, target, w2d, d2w = data.get_data(FLAGS.caption_path, FLAGS.image_path, max_len=config.num_steps+1, batch_size=config.batch_size)

    images = tf.placeholder("float", [None, 224, 224, 3], name="image")
    word = tf.placeholder(tf.int32, [None, None], name="word_seq")

    image_caption = IMAGE_ATT_CAP(images, word, None, config, is_training=True)
    config_proto = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config_proto) as sess:
        sv = load_session(sess, FLAGS.save_path)

        word = [3,1]
        words = np.array(word).reshape(1,-1)
        output,c,h = sess.run([image_caption.logits,image_caption.c,image_caption.h], feed_dict={
            image_caption.image:img, image_caption.word:words })

        print (output.shape)
        print "c:",c
        print "h:",h
        print "output:",output
        predict = output[0][-1]
        sort_idx = predict.argsort()[::-1]
        print sort_idx

        print [ d2w[p] for p in word+[sort_idx[0]]]



if __name__ == '__main__':
    train()
    #predict()

