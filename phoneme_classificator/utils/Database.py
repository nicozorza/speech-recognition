import numpy as np
from typing import List, Tuple
import pickle
from phoneme_classificator.utils.ProjectData import ProjectData
from phoneme_classificator.utils.Label import Label
from phoneme_classificator.utils.AudioFeature import AudioFeature
import tensorflow as tf
import operator



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
    def fromFile(wav_name: str, label_name: str, nfft: int, window_len: int, win_stride: int) -> 'DatabaseItem':
        # Get features
        feature = AudioFeature.fromFile(wav_name, nfft, window_len, win_stride)
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

    def __len__(self):
        return len(self.__label)


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
    def getFeatureList(self) -> List[Tuple[np.ndarray, np.ndarray, None]]:
        feature_list = []
        for _ in range(self.__length):
            feature_list.append(self.__database[_].getFeature().getFeature())
        return feature_list
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

    def getLabelsList(self) -> List[np.ndarray]:
        label_list = []
        for _ in range(self.__length):
            label_list.append(self.__database[_].getLabel().getPhonemes())
        return label_list

    def getLabelsClassesList(self) -> List[np.ndarray]:
        label_list = []
        for _ in range(self.__length):
            label_list.append(self.__database[_].getLabel().getPhonemesClass())
        return label_list

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

    def order_by_length(self):
        self.__database = sorted(self.__database, key=lambda x: len(x))

    def get_batches_list(self, batch_size) -> List['Database']:
        num_batches = int(np.ceil(len(self.__database)/batch_size))
        batch_list = []
        for _ in range(num_batches-1):
            batch_list.append(self.getRange(_*batch_size, (_+1)*batch_size))
            print(_*batch_size, (_+1)*batch_size)

        batch_list.append(self.getRange(len(self.__database)-batch_size, len(self.__database)))
        print(len(self.__database)-batch_size, len(self.__database))

        return batch_list

    def get_max_sequence_length(self):
        max_length = 0
        for _ in range(len(self.__database)):
            if len(self.__database[_]) >= max_length:
                max_length = len(self.__database[_])
        return max_length

    def pad_sequences(self):
        max_length = self.get_max_sequence_length()
        # for _ in range(len(self.__database)):
        # TODO finish this method

    def getItemFromIndex(self, index) -> DatabaseItem:
        if index > self.__length:
            return None
        return self.__database[index]

    def getRange(self, start_index, end_index) -> 'Database':
        return Database.fromList(self.__database[start_index:end_index], self.project_data)

    def __len__(self):
        return self.__length

    def save(self, file_name):

        writer = tf.python_io.TFRecordWriter(file_name)

        for _ in range(self.__length):
            feature = self.getItemFromIndex(_).getFeature().getFeature()
            label_array = np.array(self.getItemFromIndex(_).getLabel().getPhonemesClass())

            # Get feature shape
            seq_len, nfft = np.shape(feature)

            label_feat = [tf.train.Feature(int64_list=tf.train.Int64List(value=label_array))]
            feats_list = [tf.train.Feature(float_list=tf.train.FloatList(value=frame)) for frame in feature]

            feat_dict = {"feature": tf.train.FeatureList(feature=feats_list),
                         "label": tf.train.FeatureList(feature=label_feat)}

            sequence_feats = tf.train.FeatureLists(feature_list=feat_dict)

            # Context features for the entire sequence
            seq_len_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=[seq_len]))
            nfft_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=[nfft]))

            context_feats = tf.train.Features(feature={"seq_len": seq_len_feat, "nfft": nfft_feat})

            tfrecord_item = tf.train.SequenceExample(context=context_feats, feature_lists=sequence_feats)

            writer.write(tfrecord_item.SerializeToString())
        writer.close()

    @staticmethod
    def fromFile(filename: str, project_data: ProjectData) -> 'Database':

        database = Database(project_data)

        with tf.Session() as sess:
            record_iterator = tf.python_io.tf_record_iterator(path=filename)
            for record in record_iterator:

                context_features = {
                    "seq_len": tf.FixedLenFeature([], dtype=tf.int64),
                    "nfft": tf.FixedLenFeature([], dtype=tf.int64)
                }
                sequence_features = {
                    "feature": tf.VarLenFeature(dtype=tf.float32),
                    "label": tf.VarLenFeature(dtype=tf.int64)
                }

                context_parsed, sequence_parsed = tf.parse_single_sequence_example(
                    serialized=record,
                    context_features=context_features,
                    sequence_features=sequence_features
                )

                seq_len = sess.run(context_parsed["seq_len"])
                nfft = sess.run(context_parsed["nfft"])

                feature_array = sess.run(sequence_parsed["feature"])[1]
                feature_matrix = feature_array.reshape((seq_len, nfft))
                feature = AudioFeature.fromFeature(feature_matrix, nfft)

                label_array = sess.run(sequence_parsed["label"])[1]
                label = Label.fromClassArray(label_array)

                database.append(DatabaseItem(feature, label))

        return database

    @staticmethod
    def fromList(input_list: List[DatabaseItem], projectData: ProjectData) -> 'Database':
        database = Database(projectData)
        for _ in range(len(input_list)):
            database.append(input_list[_])

        return database
