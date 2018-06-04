from typing import List
from tensorflow.python.training.optimizer import Optimizer


class NetworkData:
    def __init__(self):

        self.checkpoint_path: str = None
        self.model_path: str = None

        self.num_features: int = None
        self.num_classes: int = None

        self.num_cell_units: List[int] = None

        self.keep_dropout: float = None

        self.num_dense_layers: int = None
        self.num_dense_units: List[int] = list()
        self.dense_activations: List[int] = list()
        self.dense_regularizers_beta: float = None
        self.dense_regularizers: List[int] = list()

        self.out_activation = None
        self.out_regularizer_beta: float = None
        self.out_regularizer = None

        self.optimizer: Optimizer = None

        self.learning_rate: float = None
        self.adam_epsilon: float = None

        # self.layers: List[LayerData] = list()
        # self.cost: Callable[[Any, Any], Any] = None
        # self.metrics: List[Callable[[Any, Any], Any]] = list()
        # self.optimizer: Optimizer = None
