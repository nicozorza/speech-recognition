import tensorflow as tf
from SimpleNetwork.NetworkStructure import NetworkData
from SimpleNetwork.NeuralNetwork import NeuralNetwork
from utils.Database import Database
from utils.ProjectData import ProjectData


# Get project data
project_data = ProjectData()

# Load the database
database = Database.fromFile(project_data.DATABASE_FILE)

train_inputs = database.getMfccArray()
test_inputs = train_inputs
# Readings targets
train_targets = database.getLabelsArray()
test_targets = train_targets

# THE MAIN CODE!
network_data = NetworkData()
network_data.num_features = project_data.n_mfcc
network_data.num_hidden = 100
network_data.num_layers = 1
network_data.num_classes = ord('z') - ord('a') + 1 + 1 + 1
network_data.learning_rate = 0.001
network_data.batch_size = 1
network_data.decoder = tf.nn.ctc_beam_search_decoder

neural_network = NeuralNetwork(network_data, 'a', 'a')
neural_network.create_graph()

neural_network.train(train_inputs,
                     train_targets,
                     test_inputs,
                     test_targets,
                     num_epochs=200)
