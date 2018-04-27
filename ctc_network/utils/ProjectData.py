
class ProjectData:
    def __init__(self):
        self.SOURCE_DIR = '../../audio'
        self.WAV_DIR = self.SOURCE_DIR + '/wav'
        self.TRANSCRIPTION_DIR = self.SOURCE_DIR + '/transcription'
        self.DATABASE_DIR = self.SOURCE_DIR
        self.DATABASE_NAME = 'Database'
        self.DATABASE_FILE = self.DATABASE_DIR + '/' + self.DATABASE_NAME

        self.OUT_DIR = '../out'
        self.CHECKPOINT_PATH = self.OUT_DIR + '/' + 'checkpoint'
        self.MODEL_PATH = self.OUT_DIR + '/' + 'model'

        self.n_mfcc = 26  # Number of MFCC coefficients
        self.preemphasis_coeff = 0.98
        self.frame_length = 0.02  # Length of the frame window
        self.frame_stride = 0.01  # Slide of the window
        self.fft_points = 1024
        self.num_filters = 40   # Number of filters in the filterbank
        self.lowfreq = 0
        self.highfreq = None
