import os
from utils.Database import Database, DatabaseItem
import matplotlib.pyplot as plt

# Printing parameteres
show_figures = False
print_debug = False

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
                                 label_name=label_filename)
    if print_debug:
        print(
            item.getMfcc(winlen=frame_length, winstep=frame_stride, numcep=n_mfcc, nfilt=num_filters,
                         nfft=fft_points, lowfreq=0, highfreq=None, preemph=preemphasis_coeff)
        )
        print(item.getLabelIndices())
        print(item.getTranscription())

    # Add the new data to the database
    database.append(item)

    # Print first MFCC
    if wav_index == 0 and show_figures:
        plt.figure(num=figure, figsize=(2, 2))
        figure = figure + 1
        heatmap = plt.pcolor(item.getMfcc())
        plt.title(wav_names[wav_index])
        plt.draw()

    print('Wav', wav_index+1, 'completed out of', len(wav_names))

if show_figures:
    plt.show()

# Save the database into a file
database.save(OUT_DIR + '/' + OUT_FILE)

# Load the database
database2 = Database.fromFile(OUT_DIR + '/' + OUT_FILE)

if print_debug:
    print(database.getItemFromIndex(0).getLabelIndices())
    print(database2.getItemFromIndex(0).getLabelIndices())

print("Database generated")
print("Number of elements in database: " + str(len(database)))
