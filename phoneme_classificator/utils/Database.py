import numpy as np
import random
from typing import List
import pickle
from phoneme_classificator.utils.ProjectData import ProjectData
from phoneme_classificator.utils.Label import Label
from phoneme_classificator.utils.AudioFeature import AudioFeature


class DatabaseItem(Label, AudioFeature):
    def __copy__(self):
        return self

    def __init__(self, feature: AudioFeature, label: Label):
        self.__feature: AudioFeature = feature
        self.__label: Label = label

    def getFeature(self) -> AudioFeature:
        return self.__feature

    def mfcc(self,
                winlen: float,
                winstep: float,
                numcep: int,
                nfilt: int,
                nfft: int,
                lowfreq,
                highfreq,
                preemph: float) -> np.ndarray:

        return self.__feature.mfcc(
            winlen=winlen, winstep=winstep, numcep=numcep, nfilt=nfilt, nfft=nfft,
            lowfreq=lowfreq, highfreq=highfreq, preemph=preemph)

    def getLabel(self) -> Label:
        return self.__label

    def getPhonemesClass(self) -> np.ndarray:
        return self.__label.getPhonemesClass()

    def getPhonemes(self) -> np.ndarray:
        return self.__label.getPhonemes()

    @staticmethod
    def fromFile(wav_name: str, label_name: str, window_len: int, win_stride: int) -> 'DatabaseItem':
        # Get features
        feature = AudioFeature.fromFile(wav_name, window_len, win_stride)
        sampling_rate = feature.getSamplingRate()/1000
        # Get label
        label = Label.fromFile(label_name).widowedLabel(int(window_len*sampling_rate), int(win_stride*sampling_rate))
        if len(label.getPhonemes()) != len(feature.getFeature()):
            label = DatabaseItem.__adjustSize(label, len(feature.getFeature()))

        return DatabaseItem(feature, label)

    @staticmethod
    def __adjustSize(label: Label, correct_size: int) -> Label:
        actual_len = len(label.getPhonemes())
        if actual_len > correct_size:
            return Label(label.getPhonemes()[:correct_size])
        else:
            aux_label = label.getPhonemes()
            for _ in range(correct_size-actual_len):
                aux_label = np.append(aux_label, label.getPhonemes()[-1])
            return Label(aux_label)


class Database(DatabaseItem):
    def __init__(self, project_data: ProjectData):
        self.__database: List[DatabaseItem] = []
        self.__length: int = 0
        self.project_data: ProjectData = project_data

    def append(self, item: DatabaseItem):
        self.__database.append(item)
        self.__length = len(self.__database)

    # def print(self):
    #     return self.__database
    #
    # def getMfccList(self) -> List[np.ndarray]:
    #     mfcc_list = []
    #     for _ in range(self.__length):
    #         mfcc_list.append(self.__database[_].getMfcc(
    #             winlen=self.project_data.frame_length,
    #             winstep=self.project_data.frame_stride,
    #             numcep=self.project_data.n_mfcc,
    #             nfilt=self.project_data.num_filters,
    #             nfft=self.project_data.fft_points,
    #             lowfreq=self.project_data.lowfreq,
    #             highfreq=self.project_data.highfreq,
    #             preemph=self.project_data.preemphasis_coeff))
    #     return mfcc_list
    #
    # def getMfccArray(self, normalize: bool = True) -> np.ndarray:
    #     mfcc_list = self.getMfccList()
    #     mfcc_list = np.asarray(mfcc_list)
    #     # arr, seq_len = padSequences(mfcc_list)
    #     # if normalize:
    #     #     return (mfcc_list - np.mean(mfcc_list)) / np.std(mfcc_list)
    #     # else:
    #     return mfcc_list
    #
    # def getSpectrogramList(self, nfft: int = 1024) -> List[np.ndarray]:
    #     spectrogram_list = []
    #     for _ in range(self.__length):
    #         spectrogram_list.append(self.__database[_].getSpectrogram(nfft=nfft))
    #     return spectrogram_list
    #
    # def getSpectrogramArray(self, nfft: int = 1024, normalize: bool = True) -> np.ndarray:
    #     spect_list = self.getSpectrogramList(nfft=nfft)
    #     spect_list = np.asarray(spect_list)
    #     # arr, seq_len = padSequences(mfcc_list)
    #     # if normalize:
    #     #     return (mfcc_list - np.mean(mfcc_list)) / np.std(mfcc_list)
    #     # else:
    #     return spect_list
    #
    # def getLabelsList(self) -> List[Label]:
    #     label_list = []
    #     for _ in range(self.__length):
    #         label_list.append(self.__database[_].getLabel())
    #     return label_list
    #
    # def getLabelIndicesList(self) -> List[np.ndarray]:
    #     labels_list = self.getLabelsList()
    #     index_list = []
    #     for _ in range(len(labels_list)):
    #         index_list.append(labels_list[_].toIndex())
    #     return index_list
    #
    # def getLabelsArray(self) -> np.ndarray:
    #     index_list = self.getLabelIndicesList()
    #     return np.asarray(index_list)

    def getItemFromIndex(self, index) -> DatabaseItem:
        if index > self.__length:
            return None
        return self.__database[index]

    def getRange(self, start_index, end_index) -> 'Database':
        return Database.fromList(self.__database[start_index:end_index], self.project_data)

    def __len__(self):
        return self.__length

    def save(self, file_name):
        file = open(file_name, 'wb')
        pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
        file.close()

    @staticmethod
    def fromList(input_list: List[DatabaseItem], projectData: ProjectData) -> 'Database':
        database = Database(projectData)
        for _ in range(len(input_list)):
            database.append(input_list[_])

        return database

    @staticmethod
    def fromFile(file_name: str) -> 'Database':
        # Load the database
        file = open(file_name, 'rb')
        data = pickle.load(file)
        file.close()
        return data
