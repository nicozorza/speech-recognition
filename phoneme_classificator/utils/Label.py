import numpy as np
import collections


class Phoneme:

    phonemes = {
        "iy": 0,  # Start vowels
        "ih": 1,
        "eh": 2,
        "ey": 3,
        "ae": 4,
        "aa": 5,
        "aw": 6,
        "ay": 7,
        "ah": 8,
        "ao": 9,
        "oy": 10,
        "ow": 11,
        "uh": 12,
        "uw": 13,
        "ux": 14,
        "er": 15,
        "ax": 16,
        "ix": 17,
        "axr": 18,
        "ax-h": 19,  # Last vowels

        "l": 20,  # Start semivowels
        "r": 21,
        "w": 22,
        "y": 23,
        "hh": 24,
        "hv": 25,
        "el": 26,  # Last semivowels

        "m": 27,  # Start nasals
        "n": 28,
        "ng": 29,
        "em": 30,
        "en": 31,
        "eng": 32,
        "nx": 33,  # Last nasals

        "s": 34,  # Start fricatives
        "sh": 35,
        "z": 36,
        "zh": 37,
        "f": 38,
        "th": 39,
        "v": 40,
        "dh": 41,  # Last fricatives

        "jh": 42,  # Start africates
        "ch": 43,  # Last africates

        "b": 44,  # Start stops
        "d": 45,
        "g": 46,
        "p": 47,
        "t": 48,
        "k": 49,
        "dx": 50,
        "q": 51,  # Last stops

        "pau": 52,  # Start others
        "epi": 53,
        "h#": 54,
        "1": 55,
        "2": 56  # Last others
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
