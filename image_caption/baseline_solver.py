import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
import time
import os
import cPickle as pickle
from scipy import ndimage
from utils import *
#from bleu import evaluate


class CaptioningSolver(object):
    def __init__(self, model, data, val_data, **kwargs):
        """
        Required Arguments:
            - model: Show Attend and Tell caption generating model
            - data: Training data; dictionary with the following keys:
                - features: Feature vectors of shape (82783, 196, 512)
                - file_names: Image file names of shape (82783, )
                - captions: Captions of shape (400000, 17)
                - image_idxs: Indices for mapping caption to image of shape (400000, )
                - word_to_idx: Mapping dictionary from word to index
            - val_data: validation data; for print out BLEU scores for each epoch.
        Optional Arguments:
            - n_epochs: The number of epochs to run for training.
            - batch_size: Mini batch size.
            - update_rule: A string giving the name of an update rule
            - learning_rate: Learning rate; default value is 0.01.
            - print_every: Integer; training losses will be printed every print_every iterations.
            - save_every: Integer; model variables will be saved every save_every epoch.
            - pretrained_model: String; pretrained model path
            - model_path: String; model path for saving
            - test_model: String; model path for test
        """

        self.model = model
        self.data = data
        self.val_data = val_data
        self.n_epochs = kwargs.pop('n_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.update_rule = kwargs.pop('update_rule', 'adam')
        self.learning_rate = kwargs.pop('learning_rate', 0.01)
        self.print_bleu = kwargs.pop('print_bleu', False)
        self.print_every = kwargs.pop('print_every', 10)
        self.save_every = kwargs.pop('save_every', 1)
        self.log_path = kwargs.pop('log_path', '/home/tusimple/junechen/ml_data/model/log123/')
        self.model_path = kwargs.pop('model_path', '/home/tusimple/junechen/ml_data/model/model123')
        self.pretrained_model = kwargs.pop('pretrained_model', None)
        self.test_model = kwargs.pop('test_model', './model/lstm/model-1')

        # set an optimizer by update rule
        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        elif self.update_rule == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer
        elif self.update_rule == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def load_session(self, sess, checkpoint_dir, saver):
        try:
            print("Trying to restore last checkpoint ...:",checkpoint_dir)
            last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
            saver.restore(sess, save_path=last_chk_path)
            print("restore last checkpoint %s done"%checkpoint_dir)
        except Exception as e:
            print("Failed to restore checkpoint. Initializing variables instead."),e
            sess.run(tf.global_variables_initializer())
            sess.run(tf.initialize_all_variables())

    def train(self):
        # train/val dataset
        # Changed this because I keep less features than captions, see prepro
        # n_examples = self.data['captions'].shape[0]
        n_examples = 400000
        n_iters_per_epoch = int(np.ceil(float(n_examples)/self.batch_size))

        #val_features = self.val_data['features']
        #n_iters_val = int(np.ceil(float(val_features.shape[0])/self.batch_size))

        # build graphs for training model and sampling captions
        # This scope fixed things!!
        with tf.variable_scope(tf.get_variable_scope()):
            loss = self.model.build_model()

        print "build model done"

        # train op
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            grads = tf.gradients(loss, tf.trainable_variables())
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=tf.train.get_or_create_global_step())

        # summary op
        # tf.scalar_summary('batch_loss', loss)
        tf.summary.scalar('batch_loss', loss)
        for var in tf.trainable_variables():
            #tf.histogram_summary(var.op.name, var)
            tf.summary.histogram(var.op.name, var)
        for grad, var in grads_and_vars:
            #tf.histogram_summary(var.op.name+'/gradient', grad)
            tf.summary.histogram(var.op.name+'/gradient', grad)

        #summary_op = tf.merge_all_summaries()
        summary_op = tf.summary.merge_all()
        i_global_op=tf.train.get_or_create_global_step()

        print "The number of epoch: %d" %self.n_epochs
        print "Data size: %d" %n_examples
        print "Batch size: %d" %self.batch_size
        print "Iterations per epoch: %d" %n_iters_per_epoch

        config = tf.ConfigProto(allow_soft_placement = True)
        #config.gpu_options.per_process_gpu_memory_fraction=0.9
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver(max_to_keep=40)
            self.load_session(sess, self.model_path, saver)
            tf.global_variables_initializer().run()
            threads = tf.train.start_queue_runners(sess)
            #summary_writer = tf.train.SummaryWriter(self.log_path, graph=tf.get_default_graph())
            print ("log:",self.log_path)
            summary_writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())

            prev_loss = -1
            curr_loss = 0
            start_t = time.time()

            for e in range(self.n_epochs):
                for i in range(n_iters_per_epoch):
                    feed_dict = {}
                    _, l = sess.run([train_op, loss], feed_dict)
                    curr_loss += l
                    i_global=sess.run(i_global_op)

                    # write summary for tensorboard visualization
                    if i % 10 == 0:
                        summary = sess.run(summary_op, feed_dict)
                        summary_writer.add_summary(summary, i_global)

                    if i % 10 == 0:
                        print "\nTrain loss at epoch %d & iteration %d (mini-batch): %.5f, gobal i %d, learning rate %lf" %(e+1, i+1, l, i_global, self.learning_rate)
                        print "model-%s saved." %(i_global)
                        saver.save(sess, os.path.join(self.model_path, 'model'), global_step=i_global)
                        '''
                        ground_truths = captions[image_idxs == image_idxs_batch[0]]
                        decoded = decode_captions(ground_truths, self.model.idx_to_word)
                        for j, gt in enumerate(decoded):
                            print "Ground truth %d: %s" %(j+1, gt)
                        gen_caps = sess.run(generated_captions, feed_dict)
                        decoded = decode_captions(gen_caps, self.model.idx_to_word)
                        print "Generated caption: %s\n" %decoded[0]
                        '''

                print "Previous epoch loss: ", prev_loss
                print "Current epoch loss: ", curr_loss
                print "Elapsed time: ", time.time() - start_t
                prev_loss = curr_loss
                curr_loss = 0

                self.print_bleu=False
                # print out BLEU scores and file write
                if self.print_bleu:
                    all_gen_cap = np.ndarray((val_features.shape[0], 20))
                    for i in range(n_iters_val):
                        features_batch = val_features[i*self.batch_size:(i+1)*self.batch_size]
                        feed_dict = {self.model.features: features_batch}
                        gen_cap = sess.run(generated_captions, feed_dict=feed_dict)
                        all_gen_cap[i*self.batch_size:(i+1)*self.batch_size] = gen_cap

                    all_decoded = decode_captions(all_gen_cap, self.model.idx_to_word)
                    save_pickle(all_decoded, "./data/val/val.candidate.captions.pkl")
                    scores = evaluate(data_path='./data', split='val', get_scores=True)
                    write_bleu(scores=scores, path=self.model_path, epoch=e)



