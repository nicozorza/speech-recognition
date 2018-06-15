import numpy as np
import collections


class Phoneme:  # TODO group similar labels to reduce complexity

    phonemes = {
        "h#":   0,

        "iy":   1,  # Start vowels
        "ih":   2,
        "eh":   3,
        "ey":   4,
        "ae":   5,
        "aa":   6,
        "aw":   7,
        "ay":   8,
        "ah":   9,
        "ao":   10,
        "oy":   11,
        "ow":   12,
        "uh":   13,
        "uw":   14,
        "ux":   15,
        "er":   16,
        "ax":   17,
        "ix":   18,
        "axr":  19,
        "ax-h": 20,  # Last vowels

        "l":    21,  # Start semivowels
        "r":    22,
        "w":    23,
        "y":    24,
        "hh":   25,
        "hv":   26,
        "el":   27,  # Last semivowels

        "m":    28,  # Start nasals
        "n":    29,
        "ng":   30,
        "em":   31,
        "en":   32,
        "eng":  33,
        "nx":   34,  # Last nasals

        "s":    35,  # Start fricatives
        "sh":   36,
        "z":    37,
        "zh":   38,
        "f":    39,
        "th":   40,
        "v":    41,
        "dh":   42,  # Last fricatives

        "jh":   43,  # Start africates
        "ch":   44,  # Last africates

        "b":    45,  # Start stops
        "d":    46,
        "g":    47,
        "p":    48,
        "t":    49,
        "k":    50,
        "dx":   51,
        "q":    52,  # Last stops

        "epi":  53,  # Start others
        "pau":  54,
        "1":    55,
        "2":    56,  # Last others
        "kcl":  57,
        "gcl":  58,
        "tcl":  59,
        "dcl":  60,
        "bcl":  61,
        "pcl": 62,
    }

    @staticmethod
    def phonemeToClass(phoneme: str):
        return int(Phoneme().phonemes.get(phoneme, -1))

    @staticmethod
    def classToPhoneme(index: int):
        phonemes = Phoneme().phonemes
        aux = list(phonemes.keys())[list(phonemes.values()).index(index)]
        return list(phonemes.keys())[list(phonemes.values()).index(index)]


class Label:
    def __init__(self, phonemes_array: np.ndarray):
        self.__phonemes_array: np.ndarray = phonemes_array
        self.__phonemes_class_array: np.ndarray = self.arrayToClass(phonemes_array)
        self.__phonemes_windowed_array: np.ndarray = None
        self.__phonemes_windowed_class_array: np.ndarray = None

    def get_windowed_phonemes(self) -> np.ndarray:
        return self.__phonemes_windowed_array

    def get_windowed_phonemes_class(self) -> np.ndarray:
        return self.__phonemes_windowed_class_array

    # TODO Setting methods shouldn't exist. I added them to correct other error. This also needs a review
    def set_windowed_phonemes(self, array: np.ndarray):
        self.__phonemes_windowed_array = array

    def set_windowed_phonemes_class(self, array: np.ndarray):
        self.__phonemes_windowed_class_array = array

    def get_complete_phonemes(self) -> np.ndarray:
        return self.__phonemes_array

    def get_complete_phonemes_class(self) -> np.ndarray:
        return self.__phonemes_class_array

    @staticmethod
    def __parsePhonemeLabel(phoneme_line: str):
        string_splited = phoneme_line.split(" ")
        start_index = int(string_splited[0])
        stop_index = int(string_splited[1])
        phoneme = string_splited[2].replace('\n', "")
        return [phoneme for _ in range(stop_index - start_index)]

    @staticmethod
    def fromFile(filename: str, max_len: int=None) -> 'Label':
        with open(filename, 'r') as f:
            phonemes = f.readlines()
            complete_label = np.empty(0)
            for i in range(len(phonemes)):
                complete_label = np.concatenate((complete_label, Label.__parsePhonemeLabel(phonemes[i])), axis=0)

            return Label.fromPhonemesArray(complete_label, max_len)

    @staticmethod
    def fromPhonemesArray(phonemes: np.ndarray, max_len: int = None) -> 'Label':

        if max_len is not None:
            if max_len < len(phonemes):
                raise ValueError('Invalid label length.')
            else:
                pad_len = max_len - len(phonemes)

                aux_array = np.repeat('h#', pad_len)
                phonemes = np.append(phonemes, aux_array)

        return Label(phonemes)

    def arrayToClass(self, phonemes_array: np.ndarray) -> np.ndarray:
        class_array = np.empty(0, dtype=int)
        for i in range(len(phonemes_array)):
            class_array = np.append(class_array, Phoneme.phonemeToClass(phonemes_array[i]))

        return class_array

    def windowLabel(self, win_len: int, win_stride: int):
        start: int = 0
        end: int = start + win_len
        aux_array = np.empty(0)
        while end <= len(self.__phonemes_array):
            windowed = self.__phonemes_array[start:end]
            most_common_phoneme = collections.Counter(windowed).most_common(1)[0][0]
            aux_array = np.append(aux_array, most_common_phoneme)

            start = start + win_stride
            end = start + win_len

        if start < len(self.__phonemes_array):
            windowed = self.__phonemes_array[start:]
            most_common_phoneme = collections.Counter(windowed).most_common(1)[0][0]
            aux_array = np.append(aux_array, most_common_phoneme)

        self.__phonemes_windowed_array = aux_array
        self.__phonemes_windowed_class_array = self.arrayToClass(aux_array)

    @staticmethod
    def fromClassArray(array: np.ndarray) -> 'Label':
        phoneme_array = np.empty(0, dtype=str)
        for i in range(len(array)):
            phoneme_array = np.append(phoneme_array, Phoneme.classToPhoneme(array[i]))

        return Label(phoneme_array)

    def __len__(self):
        return len(self.__phonemes_array)
