import tensorflow as tf
from phoneme_classificator.NeuralNetwork.NetworkData import NetworkData
from phoneme_classificator.NeuralNetwork.RNN import RNNClass
from phoneme_classificator.utils.Database import Database
from phoneme_classificator.utils.ProjectData import ProjectData
import numpy as np

# Load project data
project_data = ProjectData()
network_data = NetworkData()
network_data.model_path = project_data.MODEL_PATH
network_data.checkpoint_path = project_data.CHECKPOINT_PATH
network_data.num_classes = 63
network_data.keep_dropout = 1
network_data.num_features = 13
network_data.num_cell_units = [256, 128, 64]
network_data.keep_dropout = None
network_data.num_dense_layers = 3
network_data.num_dense_units = [100, 200, 100]
network_data.dense_activations = [tf.nn.tanh] * network_data.num_dense_layers
network_data.out_activation = tf.nn.relu
network_data.learning_rate = 0.001
network_data.adam_epsilon = 0.001
network_data.optimizer = tf.train.AdamOptimizer(learning_rate=network_data.learning_rate,
                                                epsilon=network_data.adam_epsilon)

network = RNNClass(network_data)
network.create_graph()

database = Database.fromFile(project_data.DATABASE_FILE, project_data)

train_feats = database.getFeatureList()
train_feats = [np.reshape(feature, [1, len(feature), network_data.num_features]) for feature in train_feats]
train_label = database.getLabelsClassesList()


network.train(
    train_features=train_feats,
    train_labels=train_label,
    training_epochs=1,
    batch_size=1
)

test_index = 10
aux = network.predict(train_feats[test_index])
print(aux)
print(train_label[test_index])
print('Accuracy: ' + str(np.mean(np.equal(aux, train_label[test_index]))))
