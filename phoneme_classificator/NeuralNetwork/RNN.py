import os
import random
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.python.training.saver import Saver

from phoneme_classificator.NeuralNetwork.NetworkData import NetworkData


class RNNClass:
    def __init__(self, network_data: NetworkData):  # , checkpoint_path: str, model_path: str
        self.graph: tf.Graph = tf.Graph()
        self.network_data = network_data

        self.seq_len = None
        self.num_features = None
        self.input_feature = None
        self.input_label = None
        self.input_label_one_hot = None
        self.rnn_cell = None
        self.multi_rrn_cell = None
        self.rnn_outputs = None
        self.dense_output = None
        self.output_classes = None
        self.output_one_hot = None
        self.loss = None
        self.correct = None
        self.training_op: tf.Operation = None
        self.checkpoint_saver: Saver = None
        self.merged_summary = None

    def create_graph(self):

        with self.graph.as_default():
            with tf.name_scope("input_context"):
                self.seq_len = tf.placeholder(tf.int32, name="sequence_length")
                self.num_features = tf.placeholder(tf.int32, name="num_features")

            with tf.name_scope("input_features"):
                self.input_feature = tf.placeholder(
                    dtype="float",
                    shape=[None, None, self.network_data.num_features],
                    name="input")
                tf.summary.image('feature', [tf.transpose(self.input_feature)])
            with tf.name_scope("input_labels"):
                self.input_label = tf.placeholder(
                    dtype=tf.int64,
                    shape=[None, None],
                    name="output")
                self.input_label_one_hot = tf.one_hot(self.input_label, self.network_data.num_classes, dtype=tf.int32)

            with tf.name_scope("RNN_cell"):
                self.rnn_cell = [tf.nn.rnn_cell.LSTMCell(num_units=self.network_data.num_cell_units[_],
                                                         state_is_tuple=True,
                                                         name='LSTM_{}'.format(_)
                                                         ) for _ in range(len(self.network_data.num_cell_units))]

                self.multi_rrn_cell = tf.nn.rnn_cell.MultiRNNCell(self.rnn_cell, state_is_tuple=True)

                self.rnn_outputs, _ = tf.nn.dynamic_rnn(
                    cell=self.multi_rrn_cell,
                    inputs=self.input_feature,
                    sequence_length=self.seq_len,
                    dtype=tf.float32,
                    scope="RNN_cell"
                )
                tf.summary.histogram('RNN', self.rnn_outputs)

            with tf.name_scope("dropout"):
                if self.network_data.keep_dropout is not None:
                    self.rnn_outputs = tf.nn.dropout(self.rnn_outputs, self.network_data.keep_dropout)

            with tf.name_scope("dense_layers"):
                for _ in range(self.network_data.num_dense_layers):
                    self.rnn_outputs = tf.layers.dense(
                        inputs=self.rnn_outputs,
                        units=self.network_data.num_dense_units[_],
                        activation=self.network_data.dense_activations[_],
                        kernel_regularizer=self.network_data.dense_regularizers[_],
                        name='dense_layer_{}'.format(_)
                    )
                    tf.summary.histogram('dense_layer', self.rnn_outputs)

            with tf.name_scope("dense_output"):
                self.dense_output = tf.layers.dense(
                    inputs=self.rnn_outputs,
                    units=self.network_data.num_classes,
                    activation=self.network_data.out_activation,
                    kernel_regularizer=self.network_data.out_regularizer,
                    name='dense_output'
                )
                tf.summary.histogram('dense_output', self.dense_output)

            with tf.name_scope("output_classes"):
                self.output_classes = tf.argmax(self.dense_output, 2)
                self.output_one_hot = tf.one_hot(self.output_classes, self.network_data.num_classes, dtype=tf.int32)[0]

            with tf.name_scope("loss"):
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.dense_output,
                    labels=self.input_label)
                self.loss = tf.reduce_mean(tf.reduce_sum(self.loss) / tf.reduce_sum(tf.cast(self.seq_len, tf.float32)))
                tf.summary.scalar('loss', self.loss)

            with tf.name_scope("correct"):
                self.correct = tf.cast(
                    tf.equal(self.output_classes, self.input_label), tf.int32)
                self.correct = \
                    tf.reduce_mean(tf.reduce_sum(tf.cast(self.correct, tf.float32), axis=1) / tf.reduce_sum(
                        tf.cast(self.seq_len, tf.float32)))
                tf.summary.scalar('accuracy', tf.reduce_mean(self.correct))

            # define the optimizer
            with tf.name_scope("training"):
                self.training_op = self.network_data.optimizer.minimize(self.loss)

            self.checkpoint_saver = tf.train.Saver(save_relative_paths=True)
            self.merged_summary = tf.summary.merge_all()

    def save_checkpoint(self, sess: tf.Session):
        if self.network_data.checkpoint_path is not None:
            self.checkpoint_saver.save(sess, self.network_data.checkpoint_path)
            # print('Saving checkpoint')

    def load_checkpoint(self, sess: tf.Session):
        if self.network_data.checkpoint_path is not None and tf.gfile.Exists("{}.meta".format(self.network_data.checkpoint_path)):
            self.checkpoint_saver.restore(sess, self.network_data.checkpoint_path)
            # print('Restoring checkpoint')
        else:
            session = tf.Session()
            session.run(tf.initialize_all_variables())

    def save_model(self, sess: tf.Session):
        if self.network_data.model_path is not None:
            drive, path_and_file = os.path.splitdrive(self.network_data.model_path)
            path, file = os.path.split(path_and_file)
            graph_io.write_graph(sess.graph, path, file, as_text=False)
            # print('Saving model')

    def create_batch(self, input_list, batch_size):
        num_batches = int(np.ceil(len(input_list) / batch_size))
        batch_list = []
        for _ in range(num_batches - 1):
            if (_ + 1) * batch_size < len(input_list):
                aux = input_list[_ * batch_size:(_ + 1) * batch_size]
            else:
                aux = input_list[len(input_list)-batch_size:len(input_list)]

            batch_list.append(aux)

        return batch_list

    def train(self,
              train_features,
              train_labels,
              batch_size: int,
              training_epochs: int,
              restore_run: bool = True,
              save_partial: bool = True,
              save_freq: int = 10,
              shuffle: bool=True,
              use_tensorboard: bool = False,
              tensorboard_freq: int = 50):

        with self.graph.as_default():
            sess = tf.Session(graph=self.graph)
            sess.run(tf.global_variables_initializer())

            if restore_run:
                self.load_checkpoint(sess)

            train_writer = None
            if use_tensorboard:
                if self.network_data.tensorboard_path is not None:
                    # Set up tensorboard summaries and saver
                    if tf.gfile.Exists(self.network_data.tensorboard_path + '/train') is not True:
                        tf.gfile.MkDir(self.network_data.tensorboard_path + '/train')
                    else:
                        tf.gfile.DeleteRecursively(self.network_data.tensorboard_path + '/train')

                train_writer = tf.summary.FileWriter("{}train".format(self.network_data.tensorboard_path), self.graph)
                train_writer.add_graph(sess.graph)

            for epoch in range(training_epochs):
                epoch_time = time.time()
                loss_ep = 0
                acc_ep = 0
                n_step = 0

                database = list(zip(train_features, train_labels))

                for batch in self.create_batch(database, batch_size):
                    batch_features, batch_labeles = zip(*batch)

                    feed_dict = {
                        self.input_feature: np.stack(batch_features),
                        self.seq_len: len(batch_features[0]),
                        self.num_features: self.network_data.num_features,
                        self.input_label: np.stack(batch_labeles)
                    }

                    loss, _, acc = sess.run([self.loss, self.training_op, self.correct], feed_dict=feed_dict)

                    loss_ep += loss
                    acc_ep += acc
                    n_step += 1
                loss_ep = loss_ep / n_step
                acc_ep = acc_ep / n_step

                if use_tensorboard:
                    if epoch % tensorboard_freq == 0 and self.network_data.tensorboard_path is not None:
                        random_index = random.randint(0, len(train_features))
                        tensorboard_feed_dict = {
                            self.input_feature: train_features[random_index],
                            self.seq_len: len(train_features[random_index][0]),
                            self.num_features: self.network_data.num_features,
                            self.input_label: train_labels[random_index]
                        }
                        s = sess.run(self.merged_summary, feed_dict=tensorboard_feed_dict)
                        train_writer.add_summary(s, epoch)

                if save_partial:
                    if epoch % save_freq == 0:
                        self.save_checkpoint(sess)
                        self.save_model(sess)

                if shuffle:
                    aux_list = list(zip(train_features, train_labels))
                    random.shuffle(aux_list)
                    train_features, train_labels = zip(*aux_list)

                print("Epoch %d of %d, loss %f, acc %f, epoch time %.2fmin, ramaining time %.2fmin" %
                      (epoch + 1,
                       training_epochs,
                       loss_ep,
                       acc_ep,
                       (time.time()-epoch_time)/60,
                       (training_epochs-epoch-1)*(time.time()-epoch_time)/60))

            # save result
            self.save_checkpoint(sess)
            self.save_model(sess)

            sess.close()

    def train_validate(self,
                       train_features,
                       train_labels,
                       val_features,
                       val_labels,
                       batch_size: int,
                       training_epochs: int,
                       val_freq: int = 50,
                       restore_run: bool = True,
                       save_partial: bool = True,
                       save_freq: int = 10,
                       shuffle: bool=True,
                       use_tensorboard: bool = False,
                       tensorboard_freq: int = 50):

        with self.graph.as_default():
            sess = tf.Session(graph=self.graph)
            sess.run(tf.global_variables_initializer())

            if restore_run:
                self.load_checkpoint(sess)

            train_writer = None
            val_writer = None
            if use_tensorboard:
                if self.network_data.tensorboard_path is not None:
                    # Set up tensorboard summaries and saver
                    if tf.gfile.Exists(self.network_data.tensorboard_path + '/train') is not True:
                        tf.gfile.MkDir(self.network_data.tensorboard_path + '/train')
                    else:
                        tf.gfile.DeleteRecursively(self.network_data.tensorboard_path + '/train')
                    # Set up tensorboard summaries and saver
                    if tf.gfile.Exists(self.network_data.tensorboard_path + '/validation') is not True:
                        tf.gfile.MkDir(self.network_data.tensorboard_path + '/validation')
                    else:
                        tf.gfile.DeleteRecursively(self.network_data.tensorboard_path + '/validation')

                train_writer = tf.summary.FileWriter("{}train".format(self.network_data.tensorboard_path), self.graph)
                train_writer.add_graph(sess.graph)
                val_writer = tf.summary.FileWriter("{}validation".format(self.network_data.tensorboard_path), self.graph)
                val_writer.add_graph(sess.graph)

            val_loss = 0
            val_acc = 0
            for epoch in range(training_epochs):
                epoch_time = time.time()
                loss_ep = 0
                acc_ep = 0
                n_step = 0
                for i in range(len(train_features)):
                    feed_dict = {
                        self.input_feature: train_features[i],
                        self.seq_len: len(train_features[i][0]),
                        self.num_features: self.network_data.num_features,
                        self.input_label: train_labels[i]
                    }

                    if use_tensorboard:
                        if epoch % tensorboard_freq == 0 and self.network_data.tensorboard_path is not None:
                            random_index = random.randint(0, len(train_features))
                            tensorboard_feed_dict = {
                                self.input_feature: train_features[random_index],
                                self.seq_len: len(train_features[random_index][0]),
                                self.num_features: self.network_data.num_features,
                                self.input_label: train_labels[random_index]
                            }
                            s = sess.run(self.merged_summary, feed_dict=tensorboard_feed_dict)
                            train_writer.add_summary(s, epoch)

                    loss, _, acc = sess.run([self.loss, self.training_op, self.correct], feed_dict=feed_dict)

                    loss_ep += loss
                    acc_ep += acc
                    n_step += 1

                if epoch % val_freq == 0:       # TODO Change it to validate with all the samples
                    # rand_index = random.randrange(0, len(val_labels))
                    rand_index = 5
                    val_feat = val_features[rand_index]
                    val_label = val_labels[rand_index]
                    val_feed_dict = {
                        self.input_feature: val_feat,
                        self.seq_len: len(val_feat[0]),
                        self.num_features: self.network_data.num_features,
                        self.input_label: val_label
                    }
                    val_loss, _, val_acc = sess.run([self.loss, self.training_op, self.correct],
                                                    feed_dict=val_feed_dict)
                    if use_tensorboard:
                        s = sess.run([self.merged_summary], feed_dict=val_feed_dict)
                        val_writer.add_summary(s, epoch)

                loss_ep = loss_ep / n_step
                acc_ep = acc_ep / n_step

                if shuffle:
                    aux_list = list(zip(train_features, train_labels))
                    random.shuffle(aux_list)
                    train_features, train_labels = zip(*aux_list)

                print("Epoch %d of %d, train_loss %f, train_acc %f, val_loss %f, val_acc %f, epoch time %.2fmin, ramaining time %.2fmin" %
                      (epoch + 1,
                       training_epochs,
                       loss_ep,
                       acc_ep,
                       val_loss,
                       val_acc,
                       (time.time() - epoch_time) / 60,
                       (training_epochs - epoch - 1) * (time.time() - epoch_time) / 60))

            # save result
            self.save_checkpoint(sess)
            self.save_model(sess)

            sess.close()

    def validate(self, features, labels):
        with self.graph.as_default():
            sess = tf.Session(graph=self.graph)
            sess.run(tf.global_variables_initializer())
            self.load_checkpoint(sess)

            sample_index = 0
            acum_accuracy = 0

            database = list(zip(features, labels))
            for item in self.create_batch(database, 1):
                feature, label = zip(*item)
                feed_dict = {
                    self.input_feature: feature,
                    self.seq_len: len(features[0]),
                    self.num_features: self.network_data.num_features,
                    self.input_label: label
                }
                accuracy = sess.run(self.correct, feed_dict=feed_dict)

                print("Index %d of %d, acc %f" % (sample_index + 1, len(labels), accuracy))
                sample_index += 1
                acum_accuracy += accuracy
            print("Validation accuracy: %f" % (acum_accuracy/len(labels)))

            sess.close()

    def predict(self, feature):

        feature = np.reshape(feature, [1, len(feature), np.shape(feature)[1]])
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            self.load_checkpoint(sess)
            feed_dict = {
                self.input_feature: feature,
                self.seq_len: len(feature[0]),
                self.num_features: self.network_data.num_features,
            }

            predicted = sess.run(self.output_classes, feed_dict=feed_dict)

            sess.close()

            return predicted


