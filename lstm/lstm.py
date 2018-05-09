import data_input
import sys

import tensorflow as tf

from tensorflow.python.client import device_lib


import numpy as np

tf.flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
tf.flags.DEFINE_string("save_path", None,
                    "Model output directory.")
tf.flags.DEFINE_string("log_path", None,
                    "Model output directory.")
FLAGS = tf.flags.FLAGS


def ptb_producer(sess, raw_data, batch_size, num_steps, name=None):
  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len], [batch_size, batch_len])
    epoch_size = (batch_len - 1) // num_steps
    epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])

    return x, y

class LSTM:
    def __init__(self, sess, input, target, is_training = True, config=None):

        self.input = input
        self.target = target
        self.sess = sess

        self.global_steps = tf.Variable(0, trainable=False)  

        self.summaries = []

        embedding = tf.get_variable("embedding", [config.vocab_size, config.hidden_size], dtype=tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, self.input)
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        self.build_net(inputs, target, is_training, config)
        self.summary_op = tf.summary.merge(self.summaries)  

    def build_net(self, inputs, targets, is_training, config):
        def make_cell():
            cell=tf.contrib.rnn.LayerNormBasicLSTMCell(config.hidden_size, forget_bias=0.0, reuse=not is_training)
            #cell=tf.contrib.rnn.LSTMBlockCell(config.hidden_size, forget_bias=0.0)
            #cell=tf.contrib.rnn.BasicLSTMCell(config.hidden_size, forget_bias=0.0, state_is_tuple=True, reuse=not is_training)
            #if is_training and config.keep_prob < 1:
            #    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
            return cell
            return tf.contrib.rnn.BasicLSTMCell(self.hidden_size, forget_bias=0.0, state_is_tuple=True, reuse=not is_training)

        cell = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        self.initial_state = cell.zero_state(config.batch_size,dtype=tf.float32)
        state = self.initial_state 

        outputs = []
        with tf.variable_scope("RNN"):
            for time_step in range(config.num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])

        vocab_size=config.vocab_size

        softmax_w = tf.get_variable("softmax_w", [config.hidden_size, vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
     # Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(logits, [config.batch_size, config.num_steps, vocab_size])
        
        print ("logit shape:",logits.shape)

    # Use the contrib sequence loss and average over the batches
        loss = tf.contrib.seq2seq.sequence_loss( logits,
            targets,
            tf.ones([config.batch_size, config.num_steps], dtype=tf.float32),
            average_across_timesteps=False,
            average_across_batch=True)

        # Update the cost
        self._cost = tf.reduce_sum(loss)
        self._final_state = state

        self.lr = tf.Variable(0.0, trainable=False)
        self.new_lr = tf.Variable(0.0, trainable=False)
        self.lr_op=tf.assign(self.lr,self.new_lr)
        tvars = tf.trainable_variables()
        grads, _= tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                      config.max_grad_norm)
        for grad in grads:  
            print ("grad name:",grad, grad.op.name)
            if grad is not None:  
                self.summaries.append(tf.summary.histogram(grad.op.name + '/gradients', grad))
        #grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars), config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=self.global_steps)
            #global_step=tf.train.get_or_create_global_step())

        return 

    def assign_lr(self, lr):
        self.sess.run(self.lr_op, feed_dict={self.new_lr:lr})

    def train(self,sess, state, inputs = None, target = None):
        '''
        print ("len:",len(inputs[0]))
        print ("len:",len(inputs[1]))
        inputs = inputs[0,[1,2]]
        #inputs = inputs[range(len(inputs)),range(self.num_steps)]
        target = target[:,0:self.num_steps]
        inputs = np.array(inputs)
        target = np.array(target)
        '''
        #print ("input size:",inputs.shape)
        feed_dict={}
        for i, (c, h) in enumerate(self.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        #feed_dict[self.input] = inputs
        #feed_dict[self.target] = target

        _,cost,lr,sumary_str=sess.run([self._train_op,self._cost, self.lr, self.summary_op],feed_dict=feed_dict)
        return cost,lr,sumary_str


class config:
    '''
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000
    '''

    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000


    '''
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000
    '''

def run_epoch(sess, _input, _target, lstm, lr_decay, epoch_size, summary_writer,sv):
    summary = tf.Summary()
    epochs = 0
    costs = 0
    state=sess.run( lstm.initial_state )
    for e in range(epoch_size):
        input = sess.run(_input)
        target = sess.run(_target)
        #cost,lr = lstm.train(sess, state)
        cost,lr,summary_str = lstm.train(sess, state, input, target)
        costs += cost
        epochs += config.num_steps
        if e % 10 == 0:
            perxity=np.exp(costs / epochs)
            summary.value.add(tag='perxity', simple_value=perxity)
            summary.value.add(tag='lr', simple_value=lr_decay)
            i_global=sess.run(lstm.global_steps)
            print (sv.global_step)
            print ("pt %f epo %d global step %d" % (perxity, epochs, i_global))
            summary_writer.add_summary(summary, i_global)#write eval to tensorboard
            summary_writer.add_summary(summary_str, i_global)#write eval to tensorboard
    save_model(sess, sv, i_global)
    return perxity


def save_model(session, sv, global_step):
     if FLAGS.save_path:
        print("Saving model to %s step %d." % (FLAGS.save_path, global_step))
        sv.saver.save(session, FLAGS.save_path, global_step=global_step)

def train():
    input = data_input.gen_data(FLAGS.data_path)
    epoch_size = ((len(input) // config.batch_size) - 1) // config.num_steps
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    c = config()
    
    _input, _target = ptb_producer(sess, input, c.batch_size, c.num_steps)
    with tf.device("/gpu:0"):
        lstm = LSTM(sess, _input, _target, is_training=True, config=c) 

    #sess.run(tf.global_variables_initializer())
    #sess.run(tf.local_variables_initializer())
    #coord = tf.train.Coordinator()  
    #threads = tf.train.start_queue_runners(sess=sess, coord=coord)  
    print ("log save:",FLAGS.log_path) 
    summary_writer = tf.summary.FileWriter(FLAGS.log_path,sess.graph)

    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    config_proto = tf.ConfigProto(allow_soft_placement=True)
    with sv.managed_session(config=config_proto) as sess:
        for i in range(config.max_max_epoch):
            x_lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
            lstm.assign_lr(config.learning_rate * x_lr_decay)
            p=run_epoch(sess, _input, _target, lstm, x_lr_decay, epoch_size, summary_writer,sv)
            print ("step %d per %f" % (i,p))
        
def main():
    train()



if __name__ == '__main__':
    main()
