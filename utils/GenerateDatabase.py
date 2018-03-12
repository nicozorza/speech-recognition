import librosa
import pickle
import os
from utils.Database import Database, Label
import matplotlib.pyplot as plt
import numpy as np
import python_speech_features as features

# Printing parameteres
show_figures = False

SOURCE_DIR = '../Audio'
WAV_DIR = SOURCE_DIR+'/wav'
TRANSCRIPTION_DIR = SOURCE_DIR+'/transcription'
OUT_DIR = SOURCE_DIR
OUT_FILE = 'Database'

n_mfcc = 13                 # Number of MFCC coefficients
preemphasis_coeff = 0.98
frame_length = 0.02         # Length of the frame window
frame_stride = 0.01         # Slide of the window
fft_points = 1024
num_filters = 40            # Number of filters in the filterbank

database = Database()

figure = 0

# Get the names of each wav file in the directory
wav_names = os.listdir(WAV_DIR)
for wav_index in range(len(wav_names)):
    # Read the wav file
    signal, fs = librosa.load(WAV_DIR + '/' + wav_names[wav_index])

    # Normalize audio
    signal = signal / abs(max(signal))

    # Get the MFCCs coefficients. The size of the matrix is n_mfcc x T, so the dimensions
    # are not the same for every sample
    mfcc = features.mfcc(signal,
                         samplerate=fs,
                         winlen=frame_length,
                         winstep=frame_stride,
                         numcep=n_mfcc,
                         nfilt=num_filters,
                         nfft=fft_points,
                         lowfreq=0,
                         highfreq=None,
                         preemph=preemphasis_coeff  # Apply a pre-emphasis filter
                         )

    # Load transcription
    trascription_name = wav_names[wav_index].split(".")[0] + '.txt'
    with open(TRANSCRIPTION_DIR + '/' + trascription_name, 'r') as f:
        transcription = f.readlines()[0]    # This method assumes that the transcription is in the first line
        # Delete blanks at the begining and the end of the transcription, transform to lowercase, etc.
        transcription = ' '.join(transcription.strip().lower().split(' ')[2:]).replace('.', '')

        label = Label(transcription)    # Create Label class from transcription

    print(transcription)
    print(label.targets)
    # Delete undesired characters
    transcription = ''.join(c for c in transcription if c not in '0123456789.,-\n\r')
    transcription = transcription.strip()   # Delete blanks at the begining and the end of the transcription
    transcription = transcription.lower()   # Transform to lowercase
    # # Add the new data to the database
    # database.append(
    #     label=int(dir_name),  # The directory name is the same as the label
    #     mfcc=mfcc
    # )
    #
    # Print first MFCC
    if wav_index == 0 and show_figures:
        plt.figure(num=figure, figsize=(2, 2))
        figure = figure + 1
        heatmap = plt.pcolor(mfcc)
        plt.title(wav_names[wav_index])
        plt.draw()

    print('Wav', wav_index+1, 'completed out of', len(wav_names), 'Label: ')

if show_figures:
    plt.show()
# # Save the database into a file
# file = open(OUT_DIR + '/' + OUT_FILE, 'wb')
# # Trim the samples to a fixed length
# pickle.dump(database.print(), file)
# file.close()

print("Database generated")
print("Number of elements in database: " + str(database.length))