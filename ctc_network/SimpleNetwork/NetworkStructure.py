
class NetworkData():
    def __init__(self):
        self.num_features: int = None
        self.num_hidden: int = None
        self.num_layers: int = None
        self.num_classes: int = None
        self.learning_rate: float = None
        self.batch_size: int = None
        self.decoder = None     # Option 1 :tf.nn.ctc_greedy_decoder, Option 2: tf.nn.ctc_beam_search_decoder

        # self.layers: List[LayerData] = list()
        # self.cost: Callable[[Any, Any], Any] = None
        # self.metrics: List[Callable[[Any, Any], Any]] = list()
        # self.optimizer: Optimizer = None
