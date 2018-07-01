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

network_data.is_bidirectional = True
network_data.rnn_regularizer = 0.2
network_data.num_fw_cell_units = [64]
network_data.num_bw_cell_units = [8]
network_data.cell_fw_activation = [tf.nn.tanh] * len(network_data.num_fw_cell_units)
network_data.cell_bw_activation = [tf.nn.tanh] * len(network_data.num_bw_cell_units)

network_data.num_dense_layers = 0
network_data.num_dense_units = [100] * network_data.num_dense_layers
network_data.dense_activations = [tf.nn.tanh] * network_data.num_dense_layers
network_data.dense_regularizer = 0.2

network_data.out_activation = tf.nn.relu
network_data.out_regularizer_beta = 0.0
network_data.out_regularizer = l2_regularizer(network_data.out_regularizer_beta)

network_data.keep_dropout = None

network_data.learning_rate = 0.01
network_data.adam_epsilon = 0.01
network_data.optimizer = tf.train.AdamOptimizer(learning_rate=network_data.learning_rate,
                                                epsilon=network_data.adam_epsilon)
###########################################################################################################

network = RNNClass(network_data)
network.create_graph()

train_database = Database.fromFile(project_data.TRAIN_DATABASE_FILE, project_data)
val_database = Database.fromFile(project_data.VAL_DATABASE_FILE, project_data)

# TODO Add a different method for this
train_feats, train_labels, _, _, _, _ = train_database.get_training_sets(1.0, 0.0, 0.0)
val_feats, val_labels, _, _, _, _ = val_database.get_training_sets(1.0, 0.0, 0.0)

network.train(
    train_features=train_feats,
    train_labels=train_labels,
    restore_run=True,
    save_partial=True,
    save_freq=10,
    use_tensorboard=True,
    tensorboard_freq=10,
    training_epochs=100,
    batch_size=50
)

network.validate(val_feats, val_labels, show_partial=False)

print(network.predict(val_feats[0]))
print(val_labels[0])
