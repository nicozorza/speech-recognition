import librosa
import numpy as np
import random
import string
import python_speech_features as features
from typing import List
import pickle


class Label:
    # Constants
    SPACE_TOKEN = '<space>'
    SPACE_INDEX = 0
    FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

    def __init__(self, transcription: str):
        self.text: str = transcription
        # Delete blanks at the beginning and the end of the transcription, transform to lowercase,
        # delete numbers in the beginning, etc.
        self.targets = (' '.join(transcription.strip().lower().split(' ')[2:]).replace('.', '')).replace(' ', '  ').split(' ')
        self.indices = None

    def toIndex(self) -> np.ndarray:
        if self.indices is None:
            # Adding blank label
            index = np.hstack([self.SPACE_TOKEN if x == '' else list(x) for x in self.targets])
            # Transform char into index
            index = np.asarray([self.SPACE_INDEX if x == '<space>' else ord(x) - self.FIRST_INDEX for x in index])
            return index
        else:
            return self.indices

    def __str__(self):
        return str(self.indices)

    @staticmethod
    def fromFile(file_name: str):
        with open(file_name, 'r') as f:
            transcription = f.readlines()[0]  # This method assumes that the transcription is in the first line

            return Label(transcription)  # Create Label class from transcription


class AudioFeature:
    def __init__(self, audio: np.ndarray, fs: float, normalize_audio=True):

        if normalize_audio:
            self.audio = audio / abs(max(audio))
        else:
            self.audio = audio
        self.fs = fs

    def getMfcc(self,
                winlen: float =0.2,
                winstep: float=0.1,
                numcep: int=13,
                nfilt: int =40,
                nfft: int =1024,
                lowfreq=0,
                highfreq=None,
                preemph: float =0.98) -> np.ndarray:

        return features.mfcc(self.audio, samplerate=self.fs, winlen=winlen, winstep=winstep, numcep=numcep,
                             nfilt=nfilt, nfft=nfft, lowfreq=lowfreq, highfreq=highfreq, preemph=preemph)

    def getSpectrogram(self) -> np.ndarray:
        raise NotImplementedError("Should have implemented this")   # TODO implement spectrogram as feature

    @staticmethod
    def fromFile(wav_name: str, normalize_audio=True):
        # Read the wav file
        signal, fs = librosa.load(wav_name)
        return AudioFeature(signal, fs, normalize_audio)


class DatabaseItem(Label, AudioFeature):
    def __copy__(self):
        return self

    def __init__(self, feature: AudioFeature, label: Label):
        self.feature = feature
        self.label = label

    def getFeature(self) -> AudioFeature:
        return self.feature

    def getMfcc(self,
                winlen: float = 0.2,
                winstep: float = 0.1,
                numcep: int = 13,
                nfilt: int = 40,
                nfft: int = 1024,
                lowfreq=0,
                highfreq=None,
                preemph: float = 0.98) -> np.ndarray:

        return self.feature.getMfcc(
            winlen=winlen, winstep=winstep, numcep=numcep, nfilt=nfilt, nfft=nfft,
            lowfreq=lowfreq, highfreq=highfreq, preemph=preemph)

    def getSpectrogram(self,
                       winlen: float = 0.2,
                       winstep: float = 0.1,
                       numcep: int = 13,
                       nfilt: int = 40,
                       nfft: int = 1024,
                       lowfreq=0,
                       highfreq=None,
                       preemph: float = 0.98) -> np.ndarray:                   # TODO implement spectrogram as feature

        return self.feature.getSpectrogram()

    def getLabel(self) -> Label:
        return self.label

    def getLabelIndices(self) -> np.ndarray:
        return self.label.toIndex()

    # def __str__(self):
    #     return 'Label: '+str(self.label)+'\n' + 'Data: '+str(self.mfcc)

    @staticmethod
    def fromFile(wav_name: str, label_name: str):
        # Get label
        label = Label.fromFile(label_name)
        # Get features
        feature = AudioFeature.fromFile(wav_name)

        return DatabaseItem(feature, label)


class Database(DatabaseItem):
    def __init__(self, batch_size: int = 50):
        self.__database = []
        self.batch_size = batch_size
        self.length: int = 0

        self.batch_count = 0
        self.batch_plan = None

    def append(self, item: DatabaseItem):
        self.__database.append(item)
        self.length = len(self.__database)

    def print(self):
        return self.__database

    def getMfccList(self) -> List[np.ndarray]:
        mfcc_list = []
        for _ in range(self.length):
            mfcc_list.append(self.__database[_].getMfcc())
        return mfcc_list

    def getSpectrogramList(self) -> List[np.ndarray]:
        spectrogram_list = []
        for _ in range(self.length):
            spectrogram_list.append(self.__database[_].getSpectrogram())
        return spectrogram_list

    def getLabelsList(self) -> List[Label]:
        label_list = []
        for _ in range(self.length):
            label_list.append(self.__database[_].getLabel())
        return label_list

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
        if index > self.length:
            return None
        return self.__database[index]

    def getRange(self, start_index, end_index):
        if end_index > self.length or start_index > end_index:
            return None
        return Database.fromList(self.__database[start_index:end_index], self.batch_size)

    def __len__(self):
        return self.length

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