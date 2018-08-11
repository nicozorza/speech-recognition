import tensorflow as tf
import numpy as np

from hyperopt import hp, fmin, tpe, space_eval

from phoneme_classificator.NeuralNetwork.NetworkData import NetworkData
from phoneme_classificator.NeuralNetwork.RNN import RNNClass
from phoneme_classificator.utils.Database import Database
from phoneme_classificator.utils.ProjectData import ProjectData

from tensorflow.contrib.layers.python.layers.regularizers import l2_regularizer

project_data = ProjectData()
train_database = Database.fromFile(project_data.TRAIN_DATABASE_FILE, project_data)
val_database = Database.fromFile(project_data.VAL_DATABASE_FILE, project_data)

# TODO Add a different method for this
train_feats, train_labels, _, _, _, _ = train_database.get_training_sets(1.0, 0.0, 0.0)
val_feats, val_labels, _, _, _, _ = val_database.get_training_sets(1.0, 0.0, 0.0)

optimization_epochs = 100
validation_epochs = 200
space_optimization_evals = 10

space = {
    'input_dense': hp.choice('input_dense',
                           [
                               {
                                   'depth': 1,
                                   'size_1': hp.quniform('d1_size_1', 50, 150, 10),
                                   'size_2': 0
                               },
                               {
                                   'depth': 2,
                                   'size_1': hp.quniform('d2_size_1', 50, 150, 10),
                                   'size_2': hp.quniform('d2_size_2', 50, 150, 10)
                               }
                           ]),
    'out_dense': hp.choice('out_dense',
                           [
                               {
                                   'depth': 1,
                                   'size_1': hp.quniform('d3_size_1', 50, 150, 10),
                                   'size_2': 0
                               },
                               {
                                   'depth': 2,
                                   'size_1': hp.quniform('d4_size_1', 50, 150, 10),
                                   'size_2': hp.quniform('d4_size_2', 50, 150, 10)
                               }
                           ]),
    'rnn_cell': hp.choice('rnn_cell',
                           [
                               {
                                   'depth': 1,
                                   'fw_1': hp.quniform('fw1_size_1', 10, 250, 40),
                                   'bw_1': hp.quniform('bw1_size_1', 10, 250, 40),
                                   'fw_2': 0,
                                   'bw_2': 0,
                               },
                               {
                                   'depth': 2,
                                   'fw_1': hp.quniform('fw2_size_1', 10, 250, 40),
                                   'bw_1': hp.quniform('bw2_size_1', 10, 250, 40),
                                   'fw_2': hp.quniform('fw3_size_2', 10, 250, 40),
                                   'bw_2': hp.quniform('bw3_size_2', 10, 250, 40),
                               }
                           ]),
    'dense_regularizer': hp.uniform('dense_reg', 0, 1),
    'rnn_regularizer': hp.uniform('rnn_reg', 0, 1),
    # 'adam_epsilon': 10**hp.uniform('adam_epsilon_exp', -8, 1),
    # 'learning_rate': 10**hp.uniform('learning_rate_exp', -6, 1),
    # 'l2_beta': 10**hp.uniform('l2_beta_exp', -5, 1),
    # 'activation': hp.choice('activation',
    #                         [
    #                             tf.nn.sigmoid,
    #                             tf.nn.relu,
    #                             tf.nn.tanh,
    #                             tf.nn.softmax
    #                         ]),
    # 'keep_input': hp.uniform('dropout_keep_input', 0, 1),
    # 'keep_1': hp.uniform('dropout_keep_1', 0, 1),
    # 'keep_2': hp.uniform('dropout_keep_2', 0, 1),
    # 'optimizer': hp.choice('optimizer_strategy', ['adam', 'gd']),
    # 'no_rss': hp.quniform('no_rss', -150, 5, -80),
    # 'minibatch_size': 100,
    # 'training_epochs': 300
}


def get_network_data(args):
    print(args)

    network_data = NetworkData()
    network_data.model_path = project_data.MODEL_PATH
    network_data.checkpoint_path = project_data.CHECKPOINT_PATH
    network_data.tensorboard_path = project_data.TENSORBOARD_PATH

    network_data.num_classes = 63
    network_data.num_features = 13

    network_data.num_input_dense_layers = args['input_dense']['depth']
    if network_data.num_input_dense_layers == 1:
        network_data.num_input_dense_units = [args['input_dense']['size_1']]
    else:
        network_data.num_input_dense_units = [args['input_dense']['size_1'], args['input_dense']['size_2']]

    network_data.input_dense_activations = [tf.nn.tanh] * network_data.num_input_dense_layers
    network_data.input_batch_normalization = True

    network_data.is_bidirectional = True
    network_data.rnn_regularizer = args['rnn_regularizer']
    if args['rnn_cell']['depth'] == 1:
        network_data.num_fw_cell_units = [args['rnn_cell']['fw_1']]
        network_data.num_bw_cell_units = [args['rnn_cell']['bw_1']]
    else:
        network_data.num_fw_cell_units = [args['rnn_cell']['fw_1'], args['rnn_cell']['fw_2']]
        network_data.num_bw_cell_units = [args['rnn_cell']['bw_1'], args['rnn_cell']['bw_2']]
    network_data.cell_fw_activation = [tf.nn.tanh] * len(network_data.num_fw_cell_units)
    network_data.cell_bw_activation = [tf.nn.tanh] * len(network_data.num_bw_cell_units)

    network_data.num_dense_layers = args['out_dense']['depth']
    if network_data.num_dense_layers == 1:
        network_data.num_dense_units = [args['out_dense']['size_1']]
    else:
        network_data.num_dense_units = [args['out_dense']['size_1'], args['out_dense']['size_2']]
    network_data.dense_activations = [tf.nn.tanh] * network_data.num_dense_layers
    network_data.dense_regularizer = args['dense_regularizer']
    network_data.dense_batch_normalization = True

    network_data.out_activation = None
    network_data.out_regularizer_beta = 0.0
    network_data.out_regularizer = l2_regularizer(network_data.out_regularizer_beta)

    network_data.keep_dropout = None

    network_data.learning_rate = 0.001
    network_data.adam_epsilon = 0.005
    network_data.optimizer = tf.train.AdamOptimizer(learning_rate=network_data.learning_rate,
                                                    epsilon=network_data.adam_epsilon)

    return network_data


def objective(args):

    network_data = get_network_data(args)

    network = RNNClass(network_data)
    network.create_graph()

    network.train(
        train_features=train_feats,
        train_labels=train_labels,
        restore_run=False,
        save_partial=False,
        # save_freq=10,
        use_tensorboard=False,
        # tensorboard_freq=10,
        training_epochs=optimization_epochs,
        batch_size=50
    )

    acc, loss = network.validate(val_feats, val_labels, show_partial=False)

    return loss


best = fmin(objective, space, algo=tpe.suggest, max_evals=space_optimization_evals)

print(best)
print(space_eval(space, best))

optimized_cfg = space_eval(space, best)

optimized_net_data = get_network_data(optimized_cfg)
optimized_net = RNNClass(optimized_net_data)
optimized_net.create_graph()

optimized_net.train(
    train_features=train_feats,
    train_labels=train_labels,
    restore_run=False,
    save_partial=True,
    save_freq=10,
    use_tensorboard=True,
    tensorboard_freq=10,
    training_epochs=validation_epochs,
    batch_size=50
)

optimized_net.validate(val_feats, val_labels, show_partial=False)

print(optimized_net.predict(val_feats[0]))
print(val_labels[0])

with open(project_data.OUT_DIR+'/optimized_net_data.txt', 'w') as file:
    for key, value in optimized_cfg.items():
        file.write('%s:%s\n' % (key, value))


