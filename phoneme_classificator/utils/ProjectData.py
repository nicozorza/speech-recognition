
class ProjectData:
    def __init__(self):
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

        self.frame_length = 20  # Length of the frame window in ms
        self.frame_stride = 10  # Slide of the window in ms
        self.fft_points = 1024

