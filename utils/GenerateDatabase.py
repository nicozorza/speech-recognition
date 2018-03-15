import os
from utils.Database import Database, DatabaseItem
import matplotlib.pyplot as plt
from utils.ProjectData import ProjectData

# Printing parameteres
show_figures = False
print_debug = False

# Load project data
project_data = ProjectData()

database = Database(project_data)

figure = 0

# Get the names of each wav file in the directory
wav_names = os.listdir(project_data.WAV_DIR)
for wav_index in range(len(wav_names)):

    # Get filenames
    wav_filename = project_data.WAV_DIR + '/' + wav_names[wav_index]
    label_filename = project_data.TRANSCRIPTION_DIR + '/' + wav_names[wav_index].split(".")[0] + '.TXT'
    
    # Create database item
    item = DatabaseItem.fromFile(wav_name=wav_filename,
                                 label_name=label_filename)
    if print_debug:
        print(
            item.getMfcc(winlen=project_data.frame_length, winstep=project_data.frame_stride,
                         numcep=project_data.n_mfcc, nfilt=project_data.num_filters, nfft=project_data.fft_points,
                         lowfreq=0, highfreq=None, preemph=project_data.preemphasis_coeff)
        )
        print(project_data.n_mfcc)
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

    print('Wav: ', wav_names[wav_index], '------', wav_index+1, 'completed out of', len(wav_names))

if show_figures:
    plt.show()

# Save the database into a file
database.save(project_data.DATABASE_FILE)

if print_debug:
    # Load the database
    database2 = Database.fromFile(project_data.DATABASE_FILE)
    print(database.getItemFromIndex(0).getLabelIndices())
    print(database2.getItemFromIndex(0).getLabelIndices())
    aux = database2.getItemFromIndex(0).getMfcc(winlen=project_data.frame_length, winstep=project_data.frame_stride,
                             numcep=project_data.n_mfcc, nfilt=project_data.num_filters, nfft=project_data.fft_points,
                             lowfreq=0, highfreq=None, preemph=project_data.preemphasis_coeff)

    aux2 = database.getItemFromIndex(0).getMfcc(winlen=project_data.frame_length, winstep=project_data.frame_stride,
                             numcep=project_data.n_mfcc, nfilt=project_data.num_filters, nfft=project_data.fft_points,
                         lowfreq=0, highfreq=None, preemph=project_data.preemphasis_coeff)

print("Database generated")
print("Number of elements in database: " + str(len(database)))
