import tensorflow as tf
from phoneme_classificator.NeuralNetwork.NetworkData import NetworkData
from phoneme_classificator.NeuralNetwork.RNN import RNNClass
from phoneme_classificator.utils.Database import Database
from phoneme_classificator.utils.ProjectData import ProjectData
import numpy as np

# Load project data
project_data = ProjectData()
network_data = NetworkData()
network_data.num_classes = 63
network_data.keep_dropout = 1
network_data.num_features = 513
network_data.num_cell_units = 200
network_data.keep_dropout = None
network_data.num_dense_layers = 2
network_data.num_dense_units = [80, 100]
network_data.dense_activations = [tf.nn.tanh, tf.nn.tanh]
network_data.out_activation = tf.nn.relu
network_data.learning_rate = 0.0001
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
    training_epochs=100,
    batch_size=1
)
aux = network.predict(train_feats[0])
print(aux)
print(train_label[0])
