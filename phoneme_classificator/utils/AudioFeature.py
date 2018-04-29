import numpy as np
import python_speech_features as features
import scipy.io.wavfile as wav
from scipy import signal
from typing import Tuple


class AudioFeature:
    def __init__(self, audio: np.ndarray, fs: float, window_len: int, win_stride: int, normalize_audio=True):

        if normalize_audio:
            self.__audio = audio / abs(max(audio))
        else:
            self.__audio = audio
        self.__fs = fs
        _, _, self.__feature = self.log_specgram(window_len, win_stride)

    def getSamplingRate(self) -> float:
        return self.__fs

    def getFeature(self) -> Tuple[np.ndarray, np.ndarray, None]:
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

        return features.mfcc(self.__audio, samplerate=self.__fs, winlen=winlen, winstep=winstep, numcep=numcep,
                             nfilt=nfilt, nfft=nfft, lowfreq=lowfreq, highfreq=highfreq, preemph=preemph)

    def log_specgram(self, window_size=20, step_size=10, eps=1e-10) -> Tuple[np.ndarray, np.ndarray, None]:
        nperseg = int(round(window_size * self.__fs / 1e3))
        noverlap = int(round(step_size * self.__fs / 1e3))
        freqs, times, spec = signal.spectrogram(self.__audio,
                                                fs=self.__fs,
                                                window='hann',
                                                nperseg=nperseg,
                                                noverlap=noverlap,
                                                detrend=False)
        return freqs, times, np.log(spec.T.astype(np.float32) + eps)

    @staticmethod
    def fromFile(wav_name: str, window_len: int, win_stride: int, normalize_audio=True) -> 'AudioFeature':
        # Read the wav file
        fs, signal = wav.read(wav_name)
        return AudioFeature(signal, fs, window_len, win_stride, normalize_audio)