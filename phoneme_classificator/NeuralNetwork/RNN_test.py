import tensorflow as tf
import tensorlayer as tl
from tensorflow.contrib import *

from phoneme_classificator.NeuralNetwork.NetworkData import NetworkData
from phoneme_classificator.NeuralNetwork.RNN import RNNClass
from phoneme_classificator.utils.Database import Database
from phoneme_classificator.utils.ProjectData import ProjectData
import numpy as np


# # Load project data
# project_data = ProjectData()
# network_data = NetworkData()
# network_data.num_classes = 62
# network_data.keep_dropout = 1
# network_data.num_features = 513
# network_data.num_cell_units = 100
# network_data.keep_dropout = 1
# network_data.num_dense_layers = 2
# network_data.num_dense_units = [100, 100]
# network_data.dense_activations = [None, None]
# network_data.optimizer = tf.train.AdamOptimizer(0.01)
# network_data.learning_rate = 0.01
#
# net = RNNClass(network_data)
# net.create_graph()
#
# def parse_test(example):
#     context_features = {
#         "seq_len": tf.FixedLenFeature([], dtype=tf.int64),
#         "nfft": tf.FixedLenFeature([], dtype=tf.int64)
#     }
#     sequence_features = {
#         "feature": tf.VarLenFeature(dtype=tf.float32),
#         "label": tf.VarLenFeature(dtype=tf.int64)
#     }
#
#     context_parsed, sequence_parsed = tf.parse_single_sequence_example(
#         serialized=example,
#         context_features=context_features,
#         sequence_features=sequence_features
#     )
#
#     return sequence_parsed, context_parsed
#
#
# graph = tf.Graph()
#
# with graph.as_default():
#
#     data = tf.data.TFRecordDataset(project_data.DATABASE_FILE)
#     data = data.map(parse_test)
#     data = data.repeat().batch(1)
#     iterator = data.make_one_shot_iterator()
#     sequence_dict, context_dict = iterator.get_next()
#
#     seq_len = tf.placeholder(tf.int32, name="length")
#     num_features = tf.placeholder(tf.int32, name="num_features")
#
#     # defining placeholders
#     # input image placeholder
#     x = tf.placeholder(dtype="float", shape=[None, None, 513], name="input")
#     # input label placeholder
#     input_label = tf.placeholder(dtype=tf.int32, shape=[None], name="output")
#     y = tf.one_hot(input_label, project_data.num_classes, dtype=tf.int32)
#
#     cell = tf.contrib.rnn.LSTMCell(100, state_is_tuple=True)
#
#     rnn_outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=x, sequence_length=seq_len, dtype=tf.float32)
#
#     # drop_outputs = tf.nn.dropout(rnn_outputs, 0.5)
#     dense_output = tf.layers.dense(
#         inputs=rnn_outputs,
#         units=10
#     )
#     dense_output = tf.layers.dense(
#         inputs=dense_output,
#         units=10
#     )
#     dense_output = tf.layers.dense(
#         inputs=dense_output,
#         units=project_data.num_classes,
#         activation=tf.nn.softmax
#     )
#
#     correct = tf.cast(tf.equal(dense_output, tf.cast(y, dtype=tf.float32)), tf.int32)
#     correct = tf.reduce_sum(tf.cast(correct, tf.float32)) / tf.reduce_sum(tf.cast(seq_len, tf.float32))
#
#     loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dense_output[0], labels=input_label)
#     loss = tf.reduce_sum(loss) / tf.reduce_sum(tf.cast(seq_len, tf.float32))
#
#     train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
#
#     with tf.Session(graph=graph) as sess:
#         sess.run(tf.global_variables_initializer())
#
#         nfft_vale, seq_len_val, asd, label= sess.run([context_dict['nfft'], context_dict['seq_len'], sequence_dict["feature"].values, sequence_dict["label"].values])
#         nfft_vale = nfft_vale[0]
#         seq_len_val = seq_len_val[0]
#         asd = np.reshape(asd, [1, seq_len_val, nfft_vale])
#         aux = tf.reshape(sequence_dict["feature"].values, shape=[seq_len, nfft_vale])
#         asf = sess.run(correct, feed_dict={x: asd, seq_len: seq_len_val, num_features: nfft_vale,input_label: label})
#
#         losa = sess.run(train_step, feed_dict={x: asd, seq_len: seq_len_val, num_features: nfft_vale,input_label: label})
#         print(seq_len_val)
#         print(asf)
#         print(losa)



# Load project data
project_data = ProjectData()
network_data = NetworkData()
network_data.num_classes = 62
network_data.keep_dropout = 1
network_data.num_features = 513
network_data.num_cell_units = 100
network_data.keep_dropout = 1
network_data.num_dense_layers = 2
network_data.num_dense_units = [100, 100]
network_data.dense_activations = [None, None]
network_data.learning_rate = 0.01
network_data.optimizer = tf.train.AdamOptimizer(network_data.learning_rate)

network = RNNClass(network_data)
network.create_graph()


def parse_test(example):
    context_features = {
        "seq_len": tf.FixedLenFeature([], dtype=tf.int64),
        "nfft": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "feature": tf.VarLenFeature(dtype=tf.float32),
        "label": tf.VarLenFeature(dtype=tf.int64)
    }

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    return sequence_parsed, context_parsed


database = Database.fromFile(project_data.DATABASE_FILE, project_data)

train_feats = database.getFeatureList()
train_feats = [np.reshape(feature, [1, len(feature), 513]) for feature in train_feats]
train_label = database.getLabelsClassesList()

network.train(
    train_features=train_feats,
    train_labels=train_label,
    training_epochs=20,
    batch_size=1
)



# with network.graph.as_default():
#
#     data = tf.data.TFRecordDataset(project_data.DATABASE_FILE)
#     data = data.map(parse_test)
#     data = data.repeat().batch(1)
#     iterator = data.make_one_shot_iterator()
#     sequence_dict, context_dict = iterator.get_next()
#
#     with tf.Session(graph=network.graph) as sess:
#         sess.run(tf.initialize_all_variables())
#         for epoch in range(10):
#             loss_ep = 0
#             acc_ep = 0
#             n_step = 0
#             for i in range(len(train_feats)):
#                 feed_dict = {
#                     network.input_feature: train_feats[i],
#                     network.seq_len: len(train_feats[i]),
#                     network.num_features: network_data.num_features,
#                     network.input_label: train_label[i]
#                 }
#                 # loss, _, acc = sess.run([self.loss, self.training_op, self.correct], feed_dict=feed_dict)
#                 loss = sess.run(network.loss, feed_dict=feed_dict)
#                 loss_ep += loss
#                 # acc_ep += acc
#                 n_step += 1
#             loss_ep = loss_ep / n_step
#             acc_ep = acc_ep / n_step
#
#             print("Epoch %d of %d, loss %f, acc %f" % (epoch + 1, 10, loss_ep, acc_ep))
        # sess.run(tf.global_variables_initializer())
        #
        # nfft, seq_len, feat, label= sess.run(
        #     [context_dict['nfft'], context_dict['seq_len'], sequence_dict["feature"].values, sequence_dict["label"].values])
        # nfft = nfft[0]
        # seq_len = seq_len[0]
        # feat = np.reshape(feat, [1, seq_len, nfft])
        # aux = np.reshape(train_feats[0], [1, seq_len, nfft])
        #
        # loss = sess.run(
        #     network.loss,
        #     feed_dict={
        #         network.input_feature: aux,
        #         network.seq_len: len(train_feats[0]),
        #         network.num_features: 513,
        #         network.input_label: train_label[0]
        #     }
        # )
        # print(loss)

# tf.reset_default_graph()
#
# # Create input data
# X = np.random.randn(2, 10, 8)
#
# # The second example is of length 6
# X[1,6:] = 0
# X_lengths = [10, 6]
#
# cell = tf.nn.rnn_cell.LSTMCell(num_units=64, state_is_tuple=True)
#
# outputs, last_states = tf.nn.dynamic_rnn(
#     cell=cell,
#     dtype=tf.float64,
#     sequence_length=X_lengths,
#     inputs=X)
#
# result = tf.contrib.learn.run_n(
#     {"outputs": outputs, "last_states": last_states},
#     n=1,
#     feed_dict=None)
#
# assert result[0]["outputs"].shape == (2, 10, 64)
# print(result[0]["outputs"])
#
# # Outputs for the second example past past length 6 should be 0
# assert (result[0]["outputs"][1,7,:] == np.zeros(cell.output_size)).all()

 # Load project data
# project_data = ProjectData()
#
# dataset = Database.fromFile(project_data.DATABASE_FILE, project_data)
#
#
# np.random.seed(1)
# size = 100
# batch_size = 100
# n_steps = 45
# seq_width = 50
#
# initializer = tf.random_uniform_initializer(-1, 1)
#
# seq_input = tf.placeholder(tf.float32, [n_steps, batch_size, seq_width])
# # sequence we will provide at runtime
# early_stop = tf.placeholder(tf.int32)
# # what timestep we want to stop at
#
# inputs = [tf.reshape(i, (batch_size, seq_width)) for i in tf.split(0, n_steps, seq_input)]
# # inputs for rnn needs to be a list, each item being a timestep.
# # we need to split our input into each timestep, and reshape it because split keeps dims by default
#
# cell = tf.contrib.rnn.LSTMCell(size, seq_width, initializer=initializer)
# initial_state = cell.zero_state(batch_size, tf.float32)
# outputs, states = tf.models.rnn(cell, inputs, initial_state=initial_state, sequence_length=early_stop)
# # set up lstm
#
# iop = tf.initialize_all_variables()
# # create initialize op, this needs to be run by the session!
# session = tf.Session()
# session.run(iop)
# # actually initialize, if you don't do this you get errors about uninitialized stuff
#
# feed = {early_stop: 100, seq_input: np.random.rand(n_steps, batch_size, seq_width).astype('float32')}
# # define our feeds.
# # early_stop can be varied, but seq_input needs to match the shape that was defined earlier
#
# outs = session.run(outputs, feed_dict=feed)
# # run once
# # output is a list, each item being a single timestep. Items at t>early_stop are all 0s
# print
# type(outs)
# print
# len(outs)