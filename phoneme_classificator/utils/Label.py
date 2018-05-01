import numpy as np
import collections


class Phoneme:

    phonemes = {
        "h#":   0,  # Start others
        "epi":  1,
        "pau":  2,
        "1":    3,
        "2":    4,  # Last others

        "iy":   5,  # Start vowels
        "ih":   6,
        "eh":   7,
        "ey":   8,
        "ae":   9,
        "aa":   10,
        "aw":   11,
        "ay":   12,
        "ah":   13,
        "ao":   14,
        "oy":   15,
        "ow":   16,
        "uh":   17,
        "uw":   18,
        "ux":   19,
        "er":   20,
        "ax":   21,
        "ix":   22,
        "axr":  23,
        "ax-h": 24,  # Last vowels

        "l":    25,  # Start semivowels
        "r":    26,
        "w":    27,
        "y":    28,
        "hh":   29,
        "hv":   30,
        "el":   31,  # Last semivowels

        "m":    32,  # Start nasals
        "n":    33,
        "ng":   34,
        "em":   35,
        "en":   36,
        "eng":  37,
        "nx":   38,  # Last nasals

        "s":    39,  # Start fricatives
        "sh":   40,
        "z":    41,
        "zh":   42,
        "f":    43,
        "th":   44,
        "v":    45,
        "dh":   46,  # Last fricatives

        "jh":   47,  # Start africates
        "ch":   48,  # Last africates

        "b":    49,  # Start stops
        "d":    50,
        "g":    51,
        "p":    52,
        "t":    53,
        "k":    54,
        "dx":   55,
        "q":    56  # Last stops

    }

    @staticmethod
    def phonemeToClass(phoneme: str):
        return int(Phoneme().phonemes.get(phoneme, -1))

    @staticmethod
    def classToPhoneme(index: int):
        phonemes = Phoneme().phonemes
        return list(phonemes.keys())[list(phonemes.values()).index(index)]


class Label:
    def __init__(self, phonemes_array: np.ndarray):
        self.__phonemes_array: np.ndarray = phonemes_array
        self.__phonemes_class_array: np.ndarray = self.arrayToClass(phonemes_array)

    def getPhonemes(self) -> np.ndarray:
        return self.__phonemes_array

    def getPhonemesClass(self) -> np.ndarray:
        return self.__phonemes_class_array

    @staticmethod
    def __parsePhonemeLabel(phoneme_line: str):
        string_splited = phoneme_line.split(" ")
        start_index = int(string_splited[0])
        stop_index = int(string_splited[1])
        phoneme = string_splited[2].replace('\n', "")
        return [phoneme for _ in range(stop_index - start_index)]

    @staticmethod
    def fromFile(filename: str) -> 'Label':
        with open(filename, 'r') as f:
            phonemes = f.readlines()
            complete_label = np.empty(0)
            for i in range(len(phonemes)):
                complete_label = np.concatenate((complete_label, Label.__parsePhonemeLabel(phonemes[i])), axis=0)

            return Label(complete_label)

    def arrayToClass(self, phonemes_array: np.ndarray) -> np.ndarray:
        class_array = np.empty(0, dtype=int)
        for i in range(len(phonemes_array)):
            class_array = np.append(class_array, Phoneme.phonemeToClass(phonemes_array[i]))

        return class_array

    def widowedLabel(self, win_len: int, win_stride: int) -> 'Label':
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

        return Label(aux_array)

    @staticmethod
    def fromClassArray(array: np.ndarray) -> 'Label':
        phoneme_array = np.empty(0, dtype=str)
        for i in range(len(array)):
            phoneme_array = np.append(phoneme_array, Phoneme.classToPhoneme(array[i]))

        return Label(phoneme_array)
