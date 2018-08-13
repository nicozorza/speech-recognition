import tensorflow as tf
import time
from ctc_network.NeuralNetwork.NetworkData import NetworkData
import os
from tensorflow.python.framework import graph_io
import numpy as np

from ctc_network.NeuralNetwork.DataConversion import sparseTupleFrom, padSequences, indexToStr


class NeuralNetwork:

    def __init__(self, network_data: NetworkData):
        self.graph: tf.Graph = tf.Graph()
        self.network_data: NetworkData = network_data
        self.checkpoint_path: str = network_data.checkpoint_path
        self.model_path: str = network_data.model_path
        self.inputs = None
        self.targets = None
        self.seq_len = None
        self.outputs = None
        self.cost = None
        self.optimizer = None
        self.decoded = None
        self.ler = None
        self.checkpoint_saver = None
        self.graph_def = None

    def create_graph(self) -> None:
        with self.graph.as_default():

            # Placeholder for the input feature
            with tf.name_scope("input"):
                # Has size [batch_size, max_stepsize, num_features], but the
                # batch_size and max_stepsize can vary along each step
                self.inputs = tf.placeholder(tf.float32, [None, None, self.network_data.num_features])

            # Placeholder for the transcription of the feature
            with tf.name_scope("target"):
                # Here we use sparse_placeholder that will generate a
                # SparseTensor required by ctc_loss op.
                self.targets = tf.sparse_placeholder(tf.int32)

            with tf.name_scope("seq_len"):
                # 1d array of size [batch_size]
                self.seq_len = tf.placeholder(tf.int32, [None])

            with tf.name_scope("rnn_cell"):
                # Defining the cell
                # Can be:
                #   tf.nn.rnn_cell.RNNCell
                #   tf.nn.rnn_cell.GRUCell
                cell = tf.contrib.rnn.LSTMCell(self.network_data.num_cell_units, state_is_tuple=True)

            with tf.name_scope("stacking"):
                # Stacking rnn cells
                stack = tf.contrib.rnn.MultiRNNCell([cell] * 1,
                                                    state_is_tuple=True)
            with tf.name_scope("output"):
                # The second output is the last state and we will no use that
                outputs, _ = tf.nn.dynamic_rnn(stack, self.inputs, self.seq_len, dtype=tf.float32)
                shape = tf.shape(self.inputs)
                batch_s, max_timesteps = shape[0], shape[1]

                # Reshaping to apply the same weights over the timesteps
                self.outputs = tf.reshape(outputs, [-1, self.network_data.num_dense_layers])

            with tf.name_scope("init_weights"):
                # Truncated normal with mean 0 and stdev=0.1
                # Tip: Try another initialization
                # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
                W = tf.Variable(tf.truncated_normal([self.network_data.num_dense_layers,
                                                     self.network_data.num_classes],
                                                    stddev=0.1))
                # Zero initialization
                # Tip: Is tf.zeros_initializer the same?
                b = tf.Variable(tf.constant(0., shape=[self.network_data.num_classes]))

            with tf.name_scope("projection"):
                # Doing the affine projection
                logits = tf.matmul(self.outputs, W) + b

                # Reshaping back to the original shape
                logits = tf.reshape(logits, [batch_s, -1, self.network_data.num_classes])

                # Time major
                logits = tf.transpose(logits, (1, 0, 2))

            with tf.name_scope("cost"):     # TODO add regularizer
                loss = tf.nn.ctc_loss(self.targets, logits, self.seq_len)
                self.cost = tf.reduce_mean(loss)

            with tf.name_scope("training"):
                self.optimizer = tf.train.MomentumOptimizer(
                    self.network_data.learning_rate, 0.9).minimize(self.cost)

            with tf.name_scope("decoder"):
                self.decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, self.seq_len)

            with tf.name_scope("metric"):
                # Inaccuracy: label error rate
                self.ler = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.targets))

            self.checkpoint_saver = tf.train.Saver(save_relative_paths=True)

    def create_batch(self, input_list, batch_size):
        num_batches = int(np.ceil(len(input_list) / batch_size))
        batch_list = []
        for _ in range(num_batches):
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

            train_cost = train_ler = 0

            for epoch in range(training_epochs):
                epoch_time = time.time()

                database = list(zip(train_features, train_labels))
                train_cost = train_ler = 0
                for batch in self.create_batch(database, batch_size):
                    batch_features, batch_labeles = zip(*batch)

                    # Padding input to max_time_step of this batch
                    batch_train_inputs, batch_train_seq_len = padSequences(batch_features)

                    # Converting to sparse representation so as to to feed SparseTensor input
                    batch_train_targets = sparseTupleFrom(batch_labeles)

                    feed_dict = {
                        self.inputs: batch_train_inputs,
                        self.seq_len: batch_train_seq_len,
                        self.targets: batch_train_targets
                    }

                    batch_cost, _ = sess.run([self.cost, self.optimizer], feed_dict)
                    train_cost += batch_cost * batch_size
                    train_ler += sess.run(self.ler, feed_dict=feed_dict) * batch_size

                train_cost /= len(train_features)
                train_ler /= len(train_features)


                print("Epoch %d of %d, loss %f, ler %f, epoch time %.2fmin, ramaining time %.2fmin" %
                      (epoch + 1,
                       training_epochs,
                       train_cost,
                       train_ler,
                       (time.time()-epoch_time)/60,
                       (training_epochs-epoch-1)*(time.time()-epoch_time)/60))

            # save result
            self.save_checkpoint(sess)
            self.save_model(sess)

            sess.close()

            return train_ler, train_cost

    def validate(self, features, labels, show_partial: bool=True):
        with self.graph.as_default():
            sess = tf.Session(graph=self.graph)
            sess.run(tf.global_variables_initializer())
            self.load_checkpoint(sess)

            database = list(zip(features, labels))
            val_cost = val_ler = 0
            sample_index = 0
            for item in self.create_batch(database, 1):
                feature, label = zip(*item)

                # Padding input to max_time_step of this batch
                batch_train_inputs, batch_train_seq_len = padSequences(feature)

                # Converting to sparse representation so as to to feed SparseTensor input
                batch_train_targets = sparseTupleFrom(label)
                feed_dict = {
                    self.inputs: batch_train_inputs,
                    self.seq_len: batch_train_seq_len,
                    self.targets: batch_train_targets
                }
                batch_cost = sess.run(self.cost, feed_dict)
                val_cost += batch_cost
                ler = sess.run(self.ler, feed_dict=feed_dict)
                val_ler += ler

                if show_partial:
                    print("Index %d of %d, ler %f" % (sample_index + 1, len(labels), ler))

                sample_index += 1

            val_cost /= len(features)
            val_ler /= len(features)
            print("Validation ler: %f, loss: %f" % (val_ler/len(labels), val_cost/len(labels)))

            sess.close()

            return val_ler/len(labels), val_cost/len(labels)

    def predict(self, feature):
        feature = np.reshape(feature, [1, len(feature), np.shape(feature)[1]])
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            self.load_checkpoint(sess)
            input, seq_len = padSequences(feature)
            feed_dict = {
                self.inputs: input,
                self.seq_len: seq_len,
            }

            predicted = sess.run(self.decoded, feed_dict=feed_dict)[0]

            sess.close()

            return indexToStr(predicted[1])

    def save_checkpoint(self, sess: tf.Session):
        if self.checkpoint_path is not None:
            self.checkpoint_saver.save(sess, self.checkpoint_path)

    def load_checkpoint(self, sess: tf.Session):
        if self.checkpoint_path is not None and tf.gfile.Exists("{}.meta".format(self.checkpoint_path)):
            self.checkpoint_saver.restore(sess, self.checkpoint_path)
        else:
            tf.global_variables_initializer(sess)

    def save_model(self, sess: tf.Session):
        if self.model_path is not None:
            drive, path_and_file = os.path.splitdrive(self.model_path)
            path, file = os.path.split(path_and_file)
            graph_io.write_graph(sess.graph, path, file, as_text=False)
