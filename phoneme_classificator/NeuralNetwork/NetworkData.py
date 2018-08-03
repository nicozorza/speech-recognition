from typing import List
from tensorflow.python.training.optimizer import Optimizer


class NetworkData:
    def __init__(self):

        self.checkpoint_path: str = None
        self.model_path: str = None
        self.tensorboard_path: str = None

        self.num_features: int = None
        self.num_classes: int = None

        self.num_input_dense_layers: int = None
        self.num_input_dense_units: List[int] = list()
        self.input_dense_activations: List[int] = list()

        self.is_bidirectional: bool = False

        self.num_cell_units: List[int] = None
        self.rnn_regularizer: float = 0
        self.cell_activation: List[int] = list()
        self.num_fw_cell_units: List[int] = None
        self.num_bw_cell_units: List[int] = None
        self.cell_fw_activation: List[int] = list()
        self.cell_bw_activation: List[int] = list()

        self.keep_dropout: float = None

        self.num_dense_layers: int = None
        self.num_dense_units: List[int] = list()
        self.dense_activations: List[int] = list()

        self.dense_regularizer: float = None

        self.out_activation = None
        self.out_regularizer_beta: float = None
        self.out_regularizer = None

        self.use_batch_normalization: bool = False

        self.optimizer: Optimizer = None

        self.learning_rate: float = None
        self.adam_epsilon: float = None

        # self.layers: List[LayerData] = list()
        # self.cost: Callable[[Any, Any], Any] = None
        # self.metrics: List[Callable[[Any, Any], Any]] = list()
        # self.optimizer: Optimizer = None
