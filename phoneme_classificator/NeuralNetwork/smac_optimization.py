import tensorflow as tf
import numpy as np

from phoneme_classificator.NeuralNetwork.NetworkData import NetworkData
from phoneme_classificator.NeuralNetwork.RNN import RNNClass
from phoneme_classificator.utils.Database import Database
from phoneme_classificator.utils.ProjectData import ProjectData

from ConfigSpace.conditions import InCondition
from smac.configspace import ConfigurationSpace
from smac.facade.smac_facade import SMAC
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario

from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, \
    UniformIntegerHyperparameter
from smac.stats.stats import Stats

from tensorflow.contrib.layers.python.layers.regularizers import l2_regularizer


project_data = ProjectData()
train_database = Database.fromFile(project_data.TRAIN_DATABASE_FILE, project_data)
val_database = Database.fromFile(project_data.VAL_DATABASE_FILE, project_data)

# TODO Add a different method for this
train_feats, train_labels, _, _, _, _ = train_database.get_training_sets(1.0, 0.0, 0.0)
val_feats, val_labels, _, _, _, _ = val_database.get_training_sets(1.0, 0.0, 0.0)

optimization_epochs = 100
validation_epochs = 200
space_optimization_evals = 40

space = {
    'input_dense_depth': CategoricalHyperparameter("input_dense_depth", ["1", "2"], default_value="1"),
    'input_dense_1': UniformIntegerHyperparameter("input_dense_1", 50, 250, default_value=100),
    'input_dense_2': UniformIntegerHyperparameter("input_dense_2", 50, 250, default_value=100),

    'out_dense_depth': CategoricalHyperparameter("out_dense_depth", ["1", "2"], default_value="1"),
    'out_dense_1': UniformIntegerHyperparameter("out_dense_1", 50, 250, default_value=100),
    'out_dense_2': UniformIntegerHyperparameter("out_dense_2", 50, 250, default_value=100),

    'rnn_depth': CategoricalHyperparameter("rnn_depth", ["1", "2"], default_value="1"),
    'fw_1': UniformIntegerHyperparameter("fw_1", 10, 250, default_value=10),
    'fw_2': UniformIntegerHyperparameter("fw_2", 10, 250, default_value=10),
    'bw_1': UniformIntegerHyperparameter("bw_1", 10, 250, default_value=100),
    'bw_2': UniformIntegerHyperparameter("bw_2", 10, 250, default_value=100),

    'dense_regularizer': UniformFloatHyperparameter("dense_regularizer", 0, 1, default_value=0.0),
    'rnn_regularizer': UniformFloatHyperparameter("rnn_regularizer", 0, 1, default_value=0.0)
}


hyper_space_conditions = [
    InCondition(child=space['input_dense_2'], parent=space['input_dense_depth'], values=["2"]),
    InCondition(child=space['out_dense_2'], parent=space['out_dense_depth'], values=["2"]),
    InCondition(child=space['fw_2'], parent=space['rnn_depth'], values=["2"]),
    InCondition(child=space['bw_2'], parent=space['rnn_depth'], values=["2"])
]

counter = None

def get_network_data(args):

    print(args)

    network_data = NetworkData()
    network_data.model_path = project_data.MODEL_PATH
    network_data.checkpoint_path = project_data.CHECKPOINT_PATH
    network_data.tensorboard_path = project_data.TENSORBOARD_PATH

    network_data.num_classes = 63
    network_data.num_features = 26

    network_data.num_input_dense_layers = int(args['input_dense_depth'])
    if network_data.num_input_dense_layers == 1:
        network_data.num_input_dense_units = [args['input_dense_1']]
    else:
        network_data.num_input_dense_units = [args['input_dense_1'], args['input_dense_2']]

    network_data.input_dense_activations = [tf.nn.tanh] * network_data.num_input_dense_layers
    network_data.input_batch_normalization = True

    network_data.is_bidirectional = True
    network_data.rnn_regularizer = args['rnn_regularizer']
    if int(args['rnn_depth']) == 1:
        network_data.num_fw_cell_units = [args['fw_1']]
        network_data.num_bw_cell_units = [args['bw_1']]
    else:
        network_data.num_fw_cell_units = [args['fw_1'], args['fw_2']]
        network_data.num_bw_cell_units = [args['bw_1'], args['bw_2']]
    network_data.cell_fw_activation = [tf.nn.tanh] * len(network_data.num_fw_cell_units)
    network_data.cell_bw_activation = [tf.nn.tanh] * len(network_data.num_bw_cell_units)

    network_data.num_dense_layers = int(args['out_dense_depth'])
    if network_data.num_dense_layers == 1:
        network_data.num_dense_units = [args['out_dense_1']]
    else:
        network_data.num_dense_units = [args['out_dense_1'], args['out_dense_2']]
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


# logger = logging.getLogger("Hyperparameter optimization")
# logging.basicConfig(level=logging.INFO)

config_space = ConfigurationSpace()
config_space.add_hyperparameters(list(space.values()))
config_space.add_conditions(hyper_space_conditions)


scenario_dict = {"run_obj": "quality",
                 "runcount-limit": space_optimization_evals,
                 "cs": config_space,
                 "deterministic": "true",
                 "output-dir": project_data.OUT_DIR + '/smac/'
                 }

scenario = Scenario(scenario_dict)
runhistory = RunHistory(aggregate_func=None)
stats = Stats(scenario)

smac = SMAC(scenario=scenario,
            runhistory=runhistory,
            stats=stats,
            rng=np.random.RandomState(42),
            tae_runner=objective)

optimized_cfg = smac.optimize()


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
    for key, value in optimized_cfg.get_dictionary().items():
        file.write('%s:%s\n' % (key, value))

