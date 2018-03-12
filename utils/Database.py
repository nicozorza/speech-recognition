import numpy as np
import random
import string

class Label:
    def __init__(self, transcription):
        self.text = transcription

        self.targets = transcription.replace(' ', '  ').split(' ')

    #def toIndex(self):


class DatabaseItem:
    def __copy__(self):
        return self

    def __init__(self, mfcc=None, label=None):
        self.mfcc = mfcc
        self.label = label
        if mfcc is None:
            self.n_mfcc = None
            self.n_frames = None
        else:
            self.n_mfcc = np.shape(mfcc)[1]
            self.n_frames = np.shape(mfcc)[0]

    def getNumMfcc(self):
        return self.n_mfcc

    def getNumFrames(self):
        return self.n_frames


    def getMfcc(self):
        return self.mfcc

    def getLabel(self):
        return self.label

    def __str__(self):
        return 'Label: '+str(self.label)+'\n' + 'Data: '+str(self.mfcc)


class Database(DatabaseItem):
    def __init__(self, mfccDatabase=None, batch_size=100):
        if mfccDatabase is None:
            self.mfccDatabase = []
        else:
            self.mfccDatabase = mfccDatabase
        self.batch_size = batch_size
        self.length = len(self.mfccDatabase)

        self.batch_count = 0
        self.batch_plan = None

    def append(self, mfcc, label):
        self.mfccDatabase.append(DatabaseItem(mfcc, label))
        self.length = len(self.mfccDatabase)

    def print(self):
        return self.mfccDatabase

    # This method assumes the size is the same for every sample
    def getNMfcc(self):
        if self.mfccDatabase is not []:
            return self.mfccDatabase[0].getNMfcc()
        else:
            return 0

    # This method assumes the size is the same for every sample
    def getNFrames(self):
        if self.mfccDatabase is not []:
            return self.mfccDatabase[0].getNFrames()
        else:
            return 0

    def getData(self):
        data = np.ndarray(
            shape=[self.length, self.getNFrames()*self.getNMfcc()]
        )
        for _ in range(self.length):
            data[_] = np.hstack(self.mfccDatabase[_].getData())
        return data

    def getLabels(self):
        labels = np.ndarray(
            shape=[self.length, 10]
        )
        for i in range(len(self.mfccDatabase)):
            aux = np.zeros(10)
            aux[self.mfccDatabase[i].getLabel()] = 1
            labels[i] = aux
        return labels

    def create_batch_plan(self):
        self.batch_plan = self.mfccDatabase
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

        data = Database(self.batch_plan[start_index:end_index], self.batch_size)

        return data.getData(), data.getLabels()    #TODO CHEACKEAR ESTO

    def getMfccFromIndex(self, index):
        if index > self.length:
            return None
        return self.mfccDatabase[index]

    def getMfccFromRange(self, start_index, end_index):
        if end_index > self.length or start_index > end_index:
            return None
        return Database(self.mfccDatabase[start_index:end_index])

    def __len__(self):
        return self.length

    def __str__(self):
        aux = ""
        for i in range(len(self.mfccDatabase)):
            aux += str(self.mfccDatabase[i]) + '\n'
        return aux