from phoneme_classificator.utils.Label import Phoneme

class ProjectData:
    def __init__(self):
        # Files data
        self.SOURCE_DIR = '../../audio'
        self.WAV_DIR = self.SOURCE_DIR + '/wav'
        self.TRANSCRIPTION_DIR = self.SOURCE_DIR + '/transcription'
        self.PHONEMES_DIR = self.SOURCE_DIR + '/phonemes'
        self.DATABASE_DIR = self.SOURCE_DIR
        self.DATABASE_NAME = 'Database.tfrecords'
        self.DATABASE_FILE = self.DATABASE_DIR + '/' + self.DATABASE_NAME

        self.OUT_DIR = '../out'
        self.CHECKPOINT_PATH = self.OUT_DIR + '/' + 'checkpoint'
        self.MODEL_PATH = self.OUT_DIR + '/' + 'model'

        # Audio processing data
        self.frame_length = 20  # Length of the frame window in ms
        self.frame_stride = 10  # Slide of the window in ms
        self.fft_points = 1024

        # Neural network data
        self.num_classes = len(Phoneme.phonemes)
        self.num_hidden = 128


