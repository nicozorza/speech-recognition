import numpy as np
import random
import python_speech_features as features
from typing import List
import pickle
import scipy.io.wavfile as wav
from scipy.signal import spectrogram
from ctc_network.utils.ProjectData import ProjectData


class Label:
    # Constants
    SPACE_TOKEN = '<space>'
    SPACE_INDEX = 0
    FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

    def __init__(self, transcription: str):
        self.__text: str = transcription
        # Delete blanks at the beginning and the end of the transcription, transform to lowercase,
        # delete numbers in the beginning, etc.
        self.__targets = (' '.join(transcription.strip().lower().split(' ')[2:]).replace('.', '')).replace(' ', '  ').split(' ')
        self.__indices = None

    def getTranscription(self) -> str:
        return self.__text

    def toIndex(self) -> np.ndarray:
        if self.__indices is None:
            # Adding blank label
            index = np.hstack([self.SPACE_TOKEN if x == '' else list(x) for x in self.__targets])
            # Transform char into index
            index = np.asarray([self.SPACE_INDEX if x == '<space>' else ord(x) - self.FIRST_INDEX for x in index])
            return index
        else:
            return self.__indices

    def __str__(self):
        return str(self.__indices)

    @staticmethod
    def fromFile(file_name: str):
        with open(file_name, 'r') as f:
            transcription = f.readlines()[0]  # This method assumes that the transcription is in the first line

            return Label(transcription)  # Create Label class from transcription


class AudioFeature:
    def __init__(self, audio: np.ndarray, fs: float, normalize_audio=True):

        if normalize_audio:
            self.__audio = audio / abs(max(audio))
        else:
            self.__audio = audio
        self.__fs = fs

    def getMfcc(self,
                winlen: float,
                winstep: float,
                numcep: int,
                nfilt: int,
                nfft: int,
                lowfreq,
                highfreq,
                preemph: float) -> np.ndarray:

        return features.mfcc(self.__audio, samplerate=self.__fs, winlen=winlen, winstep=winstep, numcep=numcep,
                             nfilt=nfilt, nfft=nfft, lowfreq=lowfreq, highfreq=highfreq, preemph=preemph)

    def getSpectrogram(self, nfft: int) -> np.ndarray:
        _1, _2, spect = spectrogram(x=self.__audio, fs=self.__fs, nfft=nfft)
        return spect    # TODO Not working properly

    @staticmethod
    def fromFile(wav_name: str, normalize_audio=True):
        # Read the wav file
        fs, signal = wav.read(wav_name)
        return AudioFeature(signal, fs, normalize_audio)


class DatabaseItem(Label, AudioFeature):
    def __copy__(self):
        return self

    def __init__(self, feature: AudioFeature, label: Label):
        self.__feature = feature
        self.__label = label

    def getFeature(self) -> AudioFeature:
        return self.__feature

    def getMfcc(self,
                winlen: float,
                winstep: float,
                numcep: int,
                nfilt: int,
                nfft: int,
                lowfreq,
                highfreq,
                preemph: float) -> np.ndarray:

        return self.__feature.getMfcc(
            winlen=winlen, winstep=winstep, numcep=numcep, nfilt=nfilt, nfft=nfft,
            lowfreq=lowfreq, highfreq=highfreq, preemph=preemph)

    def getSpectrogram(self, nfft: int = 1024) -> np.ndarray:
        return self.__feature.getSpectrogram(nfft=nfft)

    def getLabel(self) -> Label:
        return self.__label

    def getLabelIndices(self) -> np.ndarray:
        return self.__label.toIndex()

    def getTranscription(self) -> str:
        return self.__label.getTranscription()

    @staticmethod
    def fromFile(wav_name: str, label_name: str):
        # Get label
        label = Label.fromFile(label_name)
        # Get features
        feature = AudioFeature.fromFile(wav_name)

        return DatabaseItem(feature, label)


class Database(DatabaseItem):
    def __init__(self, project_data: ProjectData, batch_size: int = 50):
        self.__database: List[DatabaseItem] = []
        self.batch_size = batch_size
        self.__length: int = 0
        self.batch_count = 0
        self.batch_plan = None
        self.project_data: ProjectData = project_data

    def append(self, item: DatabaseItem):
        self.__database.append(item)
        self.__length = len(self.__database)

    def print(self):
        return self.__database

    def getMfccList(self) -> List[np.ndarray]:
        mfcc_list = []
        for _ in range(self.__length):
            mfcc_list.append(self.__database[_].getMfcc(
                winlen=self.project_data.frame_length,
                winstep=self.project_data.frame_stride,
                numcep=self.project_data.n_mfcc,
                nfilt=self.project_data.num_filters,
                nfft=self.project_data.fft_points,
                lowfreq=self.project_data.lowfreq,
                highfreq=self.project_data.highfreq,
                preemph=self.project_data.preemphasis_coeff))
        return mfcc_list

    def getMfccArray(self, normalize: bool = True) -> np.ndarray:
        mfcc_list = self.getMfccList()
        mfcc_list = np.asarray(mfcc_list)
        # arr, seq_len = padSequences(mfcc_list)
        # if normalize:
        #     return (mfcc_list - np.mean(mfcc_list)) / np.std(mfcc_list)
        # else:
        return mfcc_list

    def getSpectrogramList(self, nfft: int = 1024) -> List[np.ndarray]:
        spectrogram_list = []
        for _ in range(self.__length):
            spectrogram_list.append(self.__database[_].getSpectrogram(nfft=nfft))
        return spectrogram_list

    def getSpectrogramArray(self, nfft: int = 1024, normalize: bool = True) -> np.ndarray:
        spect_list = self.getSpectrogramList(nfft=nfft)
        spect_list = np.asarray(spect_list)
        # arr, seq_len = padSequences(mfcc_list)
        # if normalize:
        #     return (mfcc_list - np.mean(mfcc_list)) / np.std(mfcc_list)
        # else:
        return spect_list

    def getLabelsList(self) -> List[Label]:
        label_list = []
        for _ in range(self.__length):
            label_list.append(self.__database[_].getLabel())
        return label_list

    def getLabelIndicesList(self) -> List[np.ndarray]:
        labels_list = self.getLabelsList()
        index_list = []
        for _ in range(len(labels_list)):
            index_list.append(labels_list[_].toIndex())
        return index_list

    def getLabelsArray(self) -> np.ndarray:
        index_list = self.getLabelIndicesList()
        return np.asarray(index_list)

    def create_batch_plan(self):
        self.batch_plan = self.__database
        random.shuffle(self.batch_plan)
        self.batch_count = 0

    def next_batch(self):
        if self.batch_count == 0:
            self.create_batch_plan()

        start_index = self.batch_size * self.batch_count
        end_index = start_index + self.batch_size
        self.batch_count += 1
        if end_index >= len(self.batch_plan):
            end_index = len(self.batch_plan)
            start_index = end_index - self.batch_size
            self.batch_count = 0

        return self.getRange(start_index, end_index)    # TODO Check this method

    def getItemFromIndex(self, index) -> DatabaseItem:
        if index > self.__length:
            return None
        return self.__database[index]

    def getRange(self, start_index, end_index):
        if end_index > self.__length or start_index > end_index:
            return None
        return Database.fromList(self.__database[start_index:end_index], self.batch_size)

    def __len__(self):
        return self.__length

    def save(self, file_name):
        file = open(file_name, 'wb')
        pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
        file.close()

    @staticmethod
    def fromList(input_list: List[DatabaseItem], batch_size: int = 50):
        database = Database(batch_size=batch_size)
        for _ in range(len(input_list)):
            database.append(input_list[_])

        return database

    @staticmethod
    def fromFile(file_name: str):
        # Load the database
        file = open(file_name, 'rb')
        data = pickle.load(file)
        file.close()
        return data
