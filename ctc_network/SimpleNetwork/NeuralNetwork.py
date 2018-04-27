import tensorflow as tf
import tensorlayer as tl
import time
from ctc_network.SimpleNetwork.NetworkStructure import NetworkData
import os
from tensorflow.python.framework import graph_io
import numpy as np

from ctc_network.SimpleNetwork.DataConversion import sparseTupleFrom, padSequences, indexToStr


class NeuralNetwork():

    def __init__(self, network_data: NetworkData, checkpoint_path: str, model_path: str):
        self.graph: tf.Graph = tf.Graph()
        self.network_data: NetworkData = network_data
        self.checkpoint_path: str = checkpoint_path
        self.model_path: str = model_path
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
                cell = tf.contrib.rnn.LSTMCell(self.network_data.num_hidden, state_is_tuple=True)

            with tf.name_scope("stacking"):
                # Stacking rnn cells
                stack = tf.contrib.rnn.MultiRNNCell([cell] * self.network_data.num_layers,
                                                    state_is_tuple=True)
            with tf.name_scope("output"):
                # The second output is the last state and we will no use that
                outputs, _ = tf.nn.dynamic_rnn(stack, self.inputs, self.seq_len, dtype=tf.float32)
                shape = tf.shape(self.inputs)
                batch_s, max_timesteps = shape[0], shape[1]

                # Reshaping to apply the same weights over the timesteps
                self.outputs = tf.reshape(outputs, [-1, self.network_data.num_hidden])

            with tf.name_scope("init_weights"):
                # Truncated normal with mean 0 and stdev=0.1
                # Tip: Try another initialization
                # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
                W = tf.Variable(tf.truncated_normal([self.network_data.num_hidden,
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
                self.decoded, log_prob = self.network_data.decoder(logits, self.seq_len)

            with tf.name_scope("metric"):
                # Inaccuracy: label error rate
                self.ler = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.targets))

            self.checkpoint_saver = tf.train.Saver(save_relative_paths=True)

    def train(self,
              train_inputs,
              train_targets,
              test_inputs,
              test_targets,
              num_epochs=200,
              tensorboard_epoch_freq=1,
              print_epoch_freq=1,
              load_checkpoint=False,
              session: tf.Session = None
              ):

        num_batches_per_epoch = int(len(train_inputs) / self.network_data.batch_size)

        sess = session
        if session is None:
            sess = tf.Session(graph=self.graph)

        with self.graph.as_default():
            # Initializate the weights and biases
            sess.run(tf.global_variables_initializer())

            for curr_epoch in range(num_epochs):
                train_cost = train_ler = 0
                start = time.time()

                for batch in range(num_batches_per_epoch):
                    # Getting the index
                    indexes = [i % len(train_inputs) for i in range(batch * self.network_data.batch_size, (batch + 1) * self.network_data.batch_size)]

                    batch_train_inputs = train_inputs[indexes]
                    # Padding input to max_time_step of this batch
                    batch_train_inputs, batch_train_seq_len = padSequences(batch_train_inputs)

                    # Converting to sparse representation so as to to feed SparseTensor input
                    batch_train_targets = sparseTupleFrom(train_targets[indexes])

                    feed = {self.inputs: batch_train_inputs,
                            self.targets: batch_train_targets,
                            self.seq_len: batch_train_seq_len}

                    batch_cost, _ = sess.run([self.cost, self.optimizer], feed)
                    train_cost += batch_cost * self.network_data.batch_size
                    train_ler += sess.run(self.ler, feed_dict=feed) * self.network_data.batch_size

                # Shuffle the data
                shuffled_indexes = np.random.permutation(len(train_inputs))
                train_inputs = train_inputs[shuffled_indexes]
                train_targets = train_targets[shuffled_indexes]

                # Metrics mean
                train_cost /= len(train_inputs)
                train_ler /= len(train_inputs)

                log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, time = {:.3f}"
                print(log.format(curr_epoch + 1, num_epochs, train_cost, train_ler, time.time() - start))

            # Decoding all at once. Note that this isn't the best way

            # Padding input to max_time_step of this batch
            inputs, seq_len = padSequences(test_inputs)

            # Converting to sparse representation so as to to feed SparseTensor input
            targets = sparseTupleFrom(test_targets)

            feed = {self.inputs: inputs,
                    self.targets: targets,
                    self.seq_len: seq_len
                    }

            # Decoding
            d = sess.run(self.decoded[0], feed_dict=feed)
            dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=sess)

            for i, seq in enumerate(dense_decoded):
                # seq = [s for s in seq if s != -1]
                # print('Sequence %d' % i)
                # print('\t Original:\n%s' % train_targets[i])
                # print('\t Decoded:\n%s' % seq)

                str_orig = indexToStr(train_targets[i])
                str_decoded = indexToStr(dense_decoded[i])
                print('Original: %s' % str_orig)
                print('Decoded: %s' % str_decoded)
                print('---------------------------------------')


    def save_checkpoint(self, sess: tf.Session):
        if self.checkpoint_path is not None:
            self.checkpoint_saver.save(sess, self.checkpoint_path)

    def load_checkpoint(self, sess: tf.Session):
        if self.checkpoint_path is not None and tf.gfile.Exists("{}.meta".format(self.checkpoint_path)):
            self.checkpoint_saver.restore(sess, self.checkpoint_path)
        else:
            tl.layers.initialize_global_variables(sess)

    def save_model(self, sess: tf.Session):
        if self.model_path is not None:
            drive, path_and_file = os.path.splitdrive(self.model_path)
            path, file = os.path.split(path_and_file)
            graph_io.write_graph(sess.graph, path, file, as_text=False)
