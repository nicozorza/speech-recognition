import librosa
import numpy as np
import random
import string
import python_speech_features as features
from typing import List


class Label:
    # Constants
    SPACE_TOKEN = '<space>'
    SPACE_INDEX = 0
    FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

    def __init__(self, transcription: str):
        self.text: str = transcription
        self.targets = transcription.replace(' ', '  ').split(' ')
        self.indeces = self.toIndex()

    def toIndex(self):
        # Adding blank label
        index = np.hstack([self.SPACE_TOKEN if x == '' else list(x) for x in self.targets])

        # Transform char into index
        index = np.asarray([self.SPACE_INDEX if x == '<space>' else ord(x) - self.FIRST_INDEX for x in index])
        return index

    @staticmethod
    def fromFile(file_name: str):
        with open(file_name, 'r') as f:
            transcription = f.readlines()[0]  # This method assumes that the transcription is in the first line
            # Delete blanks at the begining and the end of the transcription, transform to lowercase, etc.
            transcription = ' '.join(transcription.strip().lower().split(' ')[2:]).replace('.', '')

            return Label(transcription)  # Create Label class from transcription

    def __str__(self):
        return str(self.indeces)


class DatabaseItem(Label):
    def __copy__(self):
        return self

    def __init__(self, mfcc: np.ndarray, label: Label):
        self.mfcc = mfcc
        self.label = label
        self.n_mfcc = np.shape(mfcc)[1]
        self.n_frames = np.shape(mfcc)[0]

    def getNumMfcc(self) -> int:
        return self.n_mfcc

    def getNumFrames(self) -> int:
        return self.n_frames

    def getMfcc(self) -> np.ndarray:
        return self.mfcc

    def getLabel(self) -> Label:
        return self.label

    def __str__(self):
        return 'Label: '+str(self.label)+'\n' + 'Data: '+str(self.mfcc)

    @staticmethod
    def fromFile(
            wav_name: str,
            label_name: str,
            winlen: float =0.2,
            winstep: float=0.1,
            numcep: int=13,
            nfilt: int =40,
            nfft: int =1024,
            lowfreq=0,
            highfreq=None,
            preemph: float =0.98
    ):
        # Get label
        label = Label.fromFile(label_name)
        # Read the wav file
        signal, fs = librosa.load(wav_name)
        # Normalize audio
        signal = signal / abs(max(signal))
        # Get the MFCCs coefficients. The size of the matrix is n_mfcc x T, so the dimensions
        # are not the same for every sample
        mfcc = features.mfcc(signal,
                             samplerate=fs,
                             winlen=winlen,
                             winstep=winstep,
                             numcep=numcep,
                             nfilt=nfilt,
                             nfft=nfft,
                             lowfreq=lowfreq,
                             highfreq=highfreq,
                             preemph=preemph)

        return DatabaseItem(mfcc, label)


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

    def getLabelsList(self) -> List[Label]:
        label_list = []
        for _ in range(self.length):
            label_list.append(self.__database[_].getLabel())
        return label_list

    # def create_batch_plan(self):
    #     self.batch_plan = self.mfccDatabase
    #     random.shuffle(self.batch_plan)
    #     self.batch_count = 0
    #
    # def next_batch(self):
    #     if self.batch_count == 0:
    #         self.create_batch_plan()
    #
    #     start_index = self.batch_size * self.batch_count
    #     end_index = start_index + self.batch_size
    #     self.batch_count += 1
    #     if end_index >= len(self.batch_plan):
    #         end_index = len(self.batch_plan)
    #         start_index = end_index - self.batch_size
    #         self.batch_count = 0
    #
    #     data = Database(self.batch_plan[start_index:end_index], self.batch_size)
    #
    #     return data.getData(), data.getLabels()    #TODO CHEACKEAR ESTO

    def getItemFromIndex(self, index):
        if index > self.length:
            return None
        return self.__database[index]

    def getRange(self, start_index, end_index):
        if end_index > self.length or start_index > end_index:
            return None
        return Database.fromList(self.__database[start_index:end_index], self.batch_size)

    @staticmethod
    def fromList(list, batch_size: int = 50):
        database = Database(batch_size=batch_size)
        for _ in range(len(list)):
            database.append(list[_])

        return database

    def __len__(self):
        return self.length
    