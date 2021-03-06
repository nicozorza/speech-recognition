
class ProjectData:
    def __init__(self):
        # Files data
        self.SOURCE_DIR = 'audio'
        self.WAV_DIR = self.SOURCE_DIR + '/wav'
        self.TRANSCRIPTION_DIR = self.SOURCE_DIR + '/transcriptions'
        self.PHONEMES_DIR = self.SOURCE_DIR + '/phonemes'
        self.DATABASE_DIR = self.SOURCE_DIR
        self.TRAIN_DATABASE_NAME = 'TrainDatabase'
        self.VAL_DATABASE_NAME = 'ValidationDatabase'
        self.TEST_DATABASE_NAME = 'TestDatabase'
        self.TRAIN_DATABASE_FILE = self.DATABASE_DIR + '/' + self.TRAIN_DATABASE_NAME
        self.VAL_DATABASE_FILE = self.DATABASE_DIR + '/' + self.VAL_DATABASE_NAME
        self.TEST_DATABASE_FILE = self.DATABASE_DIR + '/' + self.TEST_DATABASE_NAME

        self.OUT_DIR = 'ctc_network/out'
        self.CHECKPOINT_PATH = self.OUT_DIR + '/' + 'checkpoint/'
        self.MODEL_PATH = self.OUT_DIR + '/' + 'model/model'

        self.TENSORBOARD_PATH = self.OUT_DIR + '/' + 'tensorboard/'
