import os
import random
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
                self.input_label = tf.placeholder(
                    dtype=tf.int64,
                    shape=[None],
                    name="output")
                self.input_label_one_hot = tf.one_hot(self.input_label, self.network_data.num_classes, dtype=tf.int32)

            with tf.name_scope("RNN_cell"):

                self.rnn_cell = [tf.nn.rnn_cell.LSTMCell(num_units=n,
                                                         state_is_tuple=True
                                                         ) for n in self.network_data.num_cell_units]

                self.multi_rrn_cell = tf.nn.rnn_cell.MultiRNNCell(self.rnn_cell, state_is_tuple=True)

                self.rnn_outputs, _ = tf.nn.dynamic_rnn(
                    cell=self.multi_rrn_cell,
                    inputs=self.input_feature,
                    sequence_length=self.seq_len,
                    dtype=tf.float32
                )

            with tf.name_scope("dropout"):
                if self.network_data.keep_dropout is not None:
                    self.rnn_outputs = tf.nn.dropout(self.rnn_outputs, self.network_data.keep_dropout)

            with tf.name_scope("dense_layers"):
                for _ in range(self.network_data.num_dense_layers):
                    self.rnn_outputs = tf.layers.dense(
                        inputs=self.rnn_outputs,
                        units=self.network_data.num_dense_units[_],
                        activation=self.network_data.dense_activations[_],
                        kernel_regularizer=self.network_data.dense_regularizers[_]
                    )
            with tf.name_scope("dense_output"):
                self.dense_output = tf.layers.dense(
                    inputs=self.rnn_outputs,
                    units=self.network_data.num_classes,
                    activation=self.network_data.out_activation,
                    kernel_regularizer=self.network_data.out_regularizer
                )

            with tf.name_scope("output_classes"):
                self.output_classes = tf.argmax(self.dense_output, 2)
                self.output_one_hot = tf.one_hot(self.output_classes, self.network_data.num_classes, dtype=tf.int32)[0]

            with tf.name_scope("loss"):
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.dense_output[0],
                    labels=self.input_label)
                self.loss = tf.reduce_sum(self.loss) / tf.reduce_sum(tf.cast(self.seq_len, tf.float32))

            with tf.name_scope("correct"):
                self.correct = tf.cast(
                    tf.equal(self.output_classes, self.input_label), tf.int32)
                self.correct = \
                    tf.reduce_sum(tf.cast(self.correct, tf.float32), axis=1) / tf.reduce_sum(tf.cast(self.seq_len, tf.float32))

            # define the optimizer
            with tf.name_scope("training"):
                self.training_op = self.network_data.optimizer.minimize(self.loss)

            self.checkpoint_saver = tf.train.Saver(save_relative_paths=True)

    def save_checkpoint(self, sess: tf.Session):
        if self.network_data.checkpoint_path is not None:
            self.checkpoint_saver.save(sess, self.network_data.checkpoint_path)
            print('Saving checkpoint')

    def load_checkpoint(self, sess: tf.Session):
        if self.network_data.checkpoint_path is not None and tf.gfile.Exists("{}.meta".format(self.network_data.checkpoint_path)):
            self.checkpoint_saver.restore(sess, self.network_data.checkpoint_path)
            print('Restoring checkpoint')
        else:
            session = tf.Session()
            session.run(tf.initialize_all_variables())

    def save_model(self, sess: tf.Session):
        if self.network_data.model_path is not None:
            drive, path_and_file = os.path.splitdrive(self.network_data.model_path)
            path, file = os.path.split(path_and_file)
            graph_io.write_graph(sess.graph, path, file, as_text=False)
            print('Saving model')

    def train(self,
              train_features,
              train_labels,
              batch_size,
              training_epochs,
              shuffle=True):

        sess = tf.Session(graph=self.graph)

        with self.graph.as_default():
            sess.run(tf.global_variables_initializer())
            for epoch in range(training_epochs):
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
                    loss, _, acc = sess.run([self.loss, self.training_op, self.correct], feed_dict=feed_dict)

                    loss_ep += loss
                    acc_ep += acc
                    n_step += 1
                loss_ep = loss_ep / n_step
                acc_ep = acc_ep / n_step

                if shuffle:
                    aux_list = list(zip(train_features, train_labels))
                    random.shuffle(aux_list)
                    train_features, train_labels = zip(*aux_list)

                print("Epoch %d of %d, loss %f, acc %f" % (epoch + 1, training_epochs,  loss_ep, acc_ep))

            # save result
            self.save_checkpoint(sess)
            self.save_model(sess)

    def validate(self, features, labels):
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            self.load_checkpoint(sess)

            sample_index = 0
            acum_accuracy = 0
            for (feature, label) in zip(features, labels):
                feed_dict = {
                    self.input_feature: feature,
                    self.seq_len: len(feature[0]),
                    self.num_features: self.network_data.num_features,
                }
                predicted = sess.run(self.output_classes, feed_dict=feed_dict)
                accuracy = float(np.mean(np.equal(predicted, label)))
                print("Index %d of %d, acc %f" % (sample_index + 1, len(labels), accuracy))
                sample_index += 1
                acum_accuracy += accuracy

            print("Validation accuracy: %f" % (acum_accuracy/len(labels)))

    def predict(self, feature):

        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            self.load_checkpoint(sess)
            feed_dict = {
                self.input_feature: feature,
                self.seq_len: len(feature[0]),
                self.num_features: self.network_data.num_features,
            }

            predicted = sess.run(self.output_classes, feed_dict=feed_dict)

            return predicted


