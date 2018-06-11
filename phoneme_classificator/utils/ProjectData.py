from phoneme_classificator.utils.Label import Phoneme

class ProjectData:
    def __init__(self):
        # Files data
        self.SOURCE_DIR = 'audio'
        self.WAV_DIR = self.SOURCE_DIR + '/wav'
        self.TRANSCRIPTION_DIR = self.SOURCE_DIR + '/transcription'
        self.PHONEMES_DIR = self.SOURCE_DIR + '/phonemes'
        self.DATABASE_DIR = self.SOURCE_DIR
        self.TRAIN_DATABASE_NAME = 'TrainDatabase'
        self.VAL_DATABASE_NAME = 'ValidationDatabase'
        self.TEST_DATABASE_NAME = 'TestDatabase'
        self.TRAIN_DATABASE_FILE = self.DATABASE_DIR + '/' + self.TRAIN_DATABASE_NAME
        self.VAL_DATABASE_FILE = self.DATABASE_DIR + '/' + self.VAL_DATABASE_NAME
        self.TEST_DATABASE_FILE = self.DATABASE_DIR + '/' + self.TEST_DATABASE_NAME

        self.OUT_DIR = 'phoneme_classificator/out'
        self.CHECKPOINT_PATH = self.OUT_DIR + '/' + 'checkpoint/'
        self.MODEL_PATH = self.OUT_DIR + '/' + 'model/model'

        self.TENSORBOARD_PATH = self.OUT_DIR + '/' + 'tensorboard/'

        # Neural network data
        self.num_classes = len(Phoneme.phonemes)


