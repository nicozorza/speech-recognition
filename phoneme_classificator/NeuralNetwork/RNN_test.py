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
network_data.max_seq_len = 470
network_data.keep_dropout = 1
network_data.num_features = 513
network_data.num_cell_units = [256, 128]
network_data.keep_dropout = None
network_data.num_dense_layers = 1
network_data.num_dense_units = [100, 200]
network_data.dense_activations = [tf.nn.tanh, tf.nn.tanh]
network_data.out_activation = tf.nn.relu
network_data.learning_rate = 0.001
network_data.adam_epsilon = 0.001
network_data.optimizer = tf.train.AdamOptimizer(learning_rate=network_data.learning_rate,
                                                epsilon=network_data.adam_epsilon)

network = RNNClass(network_data)
network.create_graph()

database = Database.fromFile(project_data.DATABASE_FILE, project_data)

network.train(
    train_database=database,
    training_epochs=100,
    batch_size=5
)
aux = network.predict(database.getItemFromIndex(0).getFeature().getFeature())
print(aux)
print(database.getItemFromIndex(0).getLabel().getPhonemesClass())
