import librosa
import pickle
import os
from utils.Database import Database, DatabaseItem
import matplotlib.pyplot as plt
import numpy as np
import python_speech_features as features

# Printing parameteres
show_figures = False

SOURCE_DIR = '../audio'
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

    # Get filenames
    wav_filename = WAV_DIR + '/' + wav_names[wav_index]
    label_filename = TRANSCRIPTION_DIR + '/' + wav_names[wav_index].split(".")[0] + '.txt'
    
    # Create database item
    item = DatabaseItem.fromFile(wav_name=wav_filename,
                                 label_name=label_filename,
                                 winlen=frame_length,
                                 winstep=frame_stride,
                                 numcep=n_mfcc,
                                 nfilt=num_filters,
                                 nfft=fft_points,
                                 lowfreq=0,
                                 highfreq=None,
                                 preemph=preemphasis_coeff)
    print(item.mfcc)
    print(item.label)
    # Add the new data to the database
    database.append(item)

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
