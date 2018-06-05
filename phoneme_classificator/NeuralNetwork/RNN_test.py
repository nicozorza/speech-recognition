import tensorflow as tf
from phoneme_classificator.NeuralNetwork.NetworkData import NetworkData
from phoneme_classificator.NeuralNetwork.RNN import RNNClass
from phoneme_classificator.utils.Database import Database
from phoneme_classificator.utils.ProjectData import ProjectData
import numpy as np
from tensorflow.contrib.layers.python.layers.regularizers import l2_regularizer

###########################################################################################################
# Load project data
project_data = ProjectData()

network_data = NetworkData()
network_data.model_path = project_data.MODEL_PATH
network_data.checkpoint_path = project_data.CHECKPOINT_PATH
network_data.tensorboard_path = project_data.TENSORBOARD_PATH

network_data.num_classes = 63
network_data.num_features = 13

network_data.num_cell_units = [64]

network_data.num_dense_layers = 0
network_data.num_dense_units = []
network_data.dense_activations = [tf.nn.tanh] * network_data.num_dense_layers
network_data.dense_regularizers_beta = 0.5
network_data.dense_regularizers = [l2_regularizer(network_data.dense_regularizers_beta)]

network_data.out_activation = tf.nn.relu
network_data.out_regularizer_beta = 0.5
network_data.out_regularizer = l2_regularizer(network_data.out_regularizer_beta)

network_data.keep_dropout = 1

network_data.learning_rate = 0.01
network_data.adam_epsilon = 0.01
network_data.optimizer = tf.train.AdamOptimizer(learning_rate=network_data.learning_rate,
                                                epsilon=network_data.adam_epsilon)
###########################################################################################################

network = RNNClass(network_data)
network.create_graph()

database = Database.fromFile(project_data.DATABASE_FILE, project_data)

train_feats, train_labels, val_feats, val_labels, _, _ = database.get_training_sets(0.9, 0.1, 0.0)
network.train(
    train_features=train_feats,
    train_labels=train_labels,
    training_epochs=1,
    batch_size=1
)

network.validate(val_feats, val_labels)

print(network.predict(val_feats[0]))
print(val_labels[0])
