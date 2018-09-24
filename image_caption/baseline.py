import tensorflow as tf
from vgg19 import Vgg19
import tensorflow as tf
import numpy as np
import math

import data
import utils
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


class CaptionGenerator(object):
    def __init__(self, feature, captions, word_to_idx, dim_feature=[196, 512], dim_embed=512, dim_hidden=1024, n_time_step=16,
                  prev2out=True, ctx2out=True, alpha_c=1.0, selector=True, dropout=True):
        """
        Args:
            word_to_idx: word-to-index mapping dictionary.
            dim_feature: (optional) Dimension of vggnet19 conv5_3 feature vectors.
            dim_embed: (optional) Dimension of word embedding.
            dim_hidden: (optional) Dimension of all hidden state.
            n_time_step: (optional) Time step size of LSTM.
            prev2out: (optional) previously generated word to hidden state. (see Eq (7) for explanation)
            ctx2out: (optional) context to hidden state (see Eq (7) for explanation)
            alpha_c: (optional) Doubly stochastic regularization coefficient. (see Section (4.2.1) for explanation)
            selector: (optional) gating scalar for context vector. (see Section (4.2.1) for explanation)
            dropout: (optional) If true then dropout layer is added.
        """

        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
        self.prev2out = prev2out
        self.ctx2out = ctx2out
        self.alpha_c = alpha_c
        self.selector = selector
        self.dropout = dropout
        self.V = len(word_to_idx)
        self.L = dim_feature[0]
        self.D = dim_feature[1]
        self.M = dim_embed
        self.H = dim_hidden
        self.T = n_time_step
        self._start = word_to_idx['<start>']
        self._null = word_to_idx['<null>']

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
        self.features = feature
        self.features = tf.reshape(self.features,[-1] + dim_feature)
        self.captions = captions

    def _get_initial_lstm(self, features):
        with tf.variable_scope('initial_lstm'):
            features_mean = tf.reduce_mean(features, 1)

            w_h = tf.get_variable('w_h', [self.D, self.H], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

            w_c = tf.get_variable('w_c', [self.D, self.H], initializer=self.weight_initializer)
            b_c = tf.get_variable('b_c', [self.H], initializer=self.const_initializer)
            c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
            return c, h

    def _word_embedding(self, inputs, reuse=False):
        with tf.variable_scope('word_embedding', reuse=reuse):
            w = tf.get_variable('w', [self.V, self.M], initializer=self.emb_initializer)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return x

    def _project_features(self, features):
        with tf.variable_scope('project_features'):
            w = tf.get_variable('w', [self.D, self.D], initializer=self.weight_initializer)
            features_flat = tf.reshape(features, [-1, self.D])
            features_proj = tf.matmul(features_flat, w)
            features_proj = tf.reshape(features_proj, [-1, self.L, self.D])
            return features_proj

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

    def _selector(self, context, h, reuse=False):
        with tf.variable_scope('selector', reuse=reuse):
            w = tf.get_variable('w', [self.H, 1], initializer=self.weight_initializer)
            b = tf.get_variable('b', [1], initializer=self.const_initializer)
            beta = tf.nn.sigmoid(tf.matmul(h, w) + b, 'beta')    # (N, 1)
            context = tf.multiply(beta, context, name='selected_context')
            return context, beta

    def _decode_lstm(self, x, h, context, dropout=False, reuse=False):
        with tf.variable_scope('logits', reuse=reuse):
            w_h = tf.get_variable('w_h', [self.H, self.M], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.M], initializer=self.const_initializer)
            w_out = tf.get_variable('w_out', [self.M, self.V], initializer=self.weight_initializer)
            b_out = tf.get_variable('b_out', [self.V], initializer=self.const_initializer)

            if dropout:
                h = tf.nn.dropout(h, 0.5)
            h_logits = tf.matmul(h, w_h) + b_h

            if self.ctx2out:
                w_ctx2out = tf.get_variable('w_ctx2out', [self.D, self.M], initializer=self.weight_initializer)
                h_logits += tf.matmul(context, w_ctx2out)

            if self.prev2out:
                h_logits += x
            h_logits = tf.nn.tanh(h_logits)

            if dropout:
                h_logits = tf.nn.dropout(h_logits, 0.5)
            out_logits = tf.matmul(h_logits, w_out) + b_out
            return out_logits

    def _batch_norm(self, x, mode='train', name=None):
        return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=(mode=='train'),
                                            updates_collections=None,
                                            scope=(name+'batch_norm'))

    def build_model(self):
        features = self.features
        captions = self.captions
        batch_size = tf.shape(features)[0]

        captions_in = captions[:, :self.T]
        captions_out = captions[:, 1:]
        mask = tf.to_float(tf.not_equal(captions_out, self._null))


        # batch normalize feature vectors
        features = self._batch_norm(features, mode='train', name='conv_features')

        c, h = self._get_initial_lstm(features=features)
        x = self._word_embedding(inputs=captions_in)
        features_proj = self._project_features(features=features)

        loss = 0.0
        alpha_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)
        print ("lstm")
        for t in range(self.T):
            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t!=0))
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t!=0))

            with tf.variable_scope('lstm', reuse=(t!=0)):
                _, (c, h) = lstm_cell(inputs=tf.concat( [x[:,t,:], context],1), state=[c, h])

            logits = self._decode_lstm(x[:,t,:], h, context, dropout=self.dropout, reuse=(t!=0))

            loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=captions_out[:, t],logits=logits)*mask[:, t] )

        if self.alpha_c > 0:
            alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))     # (N, T, L)
            alphas_all = tf.reduce_sum(alphas, 1)      # (N, L)
            alpha_reg = self.alpha_c * tf.reduce_sum((16./196 - alphas_all) ** 2)
            loss += alpha_reg

        return loss / tf.to_float(batch_size)

    def build_sampler(self, max_len=20):
        features = self.features

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='test', name='conv_features')

        c, h = self._get_initial_lstm(features=features)
        features_proj = self._project_features(features=features)

        sampled_word_list = []
        alpha_list = []
        beta_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        for t in range(max_len):
            if t == 0:
                x = self._word_embedding(inputs=tf.fill([tf.shape(features)[0]], self._start))
            else:
                x = self._word_embedding(inputs=sampled_word, reuse=True)

            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t!=0))
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t!=0))
                beta_list.append(beta)

            with tf.variable_scope('lstm', reuse=(t!=0)):
                _, (c, h) = lstm_cell(inputs=tf.concat( [x, context],1), state=[c, h])

            logits = self._decode_lstm(x, h, context, reuse=(t!=0))
            sampled_word = tf.argmax(logits, 1)
            sampled_word_list.append(sampled_word)

        alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))     # (N, T, L)
        betas = tf.transpose(tf.squeeze(beta_list), (1, 0))    # (N, T)
        sampled_captions = tf.transpose(tf.stack(sampled_word_list), (1, 0))     # (N, max_len)
        return alphas, betas, sampled_captions

class img_config():
    hidden_size=1024
    vob_size=20000
    embedding_size=512
    feature_dim = [196,512]
    seq_len = 20
    num_layers = 2

    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20


def train():

    config = img_config()
    #config.batch_size=1
    f, image,label,word, target, w2d,d2w = data.get_data(FLAGS.caption_path, FLAGS.image_path, max_len=config.seq_len+1, batch_size=config.batch_size)

    with tf.name_scope("content_vgg"):
            cnn_net = Vgg19()
            cnn_net.build(image)

    image_feature = cnn_net.conv5_3 #[-1,14*14,512]

    cg = CaptionGenerator(w2d,n_time_step=config.seq_len)
    loss = cg.build_model()

    tvars = tf.trainable_variables()
    grads, _= tf.clip_by_global_norm(tf.gradients(loss, tvars), config.max_grad_norm)
    lr = tf.Variable(0.0, trainable=False)
    new_lr = tf.Variable(0.0, trainable=False)

    lr_op=tf.assign(lr,new_lr)

    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.train.get_or_create_global_step())


    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    config_proto = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    with sv.managed_session(config=config_proto) as sess:
        threads = tf.train.start_queue_runners(sess)

        summary_writer = tf.summary.FileWriter(FLAGS.log_path,sess.graph)

        for i in range(config.max_max_epoch):
            x_lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
            print ("lr:",x_lr_decay)
            _,c_lr=sess.run([lr_op,lr], feed_dict={new_lr:config.learning_rate * x_lr_decay})

            for j in range(414113/config.batch_size):
                [img_f, feature,img_label] = sess.run( [f, image_feature, label] )

                _, c_loss = sess.run( [ train_op, loss ], feed_dict={
                                cg.features : feature.reshape(-1,196,512),
                                cg.captions : img_label})
                if j % 10 == 0:
                    summary = tf.Summary()
                    summary.value.add(tag='loss', simple_value=c_loss)
                    i_global=sess.run(tf.train.get_or_create_global_step())
                    print ("cost %f global step %d" % (c_loss, i_global))
                    summary_writer.add_summary(summary, i_global)#write eval to tensorboard
                if j % 100 == 0:
                    print img_f
                    print img_label
                    print [ d2w[p] for p in img_label[0]]


def predict():

    config = img_config()

    w2d, d2w = data.get_word_to_id()

    #img = utils.load_image("/home/tusimple/junechen/ml_data/data/train2014/COCO_train2014_000000160629.jpg")
    img = utils.load_image("/home/tusimple/junechen/ml_data/data/train2014/COCO_train2014_000000318556.jpg")
    img = img.reshape((1, 224, 224, 3))

    images = tf.placeholder("float", [None, 224, 224, 3], name="image")

    with tf.name_scope("content_vgg"):
            cnn_net = Vgg19()
            cnn_net.build(images)

    with tf.device("/gpu:1"):
        image_feature = cnn_net.conv5_3 #[-1,14*14,512]

        cg = CaptionGenerator(w2d,n_time_step=config.seq_len)
        alphas, betas, sampled_captions =  cg.build_sampler()

    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    config_proto = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    with sv.managed_session(config=config_proto) as sess:
        [feature] = sess.run( [image_feature], feed_dict={ images: img } )

        [image_label] = sess.run( [sampled_captions], feed_dict={
                                cg.features : feature.reshape(-1,196,512) } )

        print image_label
        print [ d2w[p] for p in image_label[0]]



#train()
#predict()

