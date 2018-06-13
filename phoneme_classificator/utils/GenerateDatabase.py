import os

import matplotlib.pyplot as plt
import numpy as np
from phoneme_classificator.utils.AudioFeature import FeatureConfig
from phoneme_classificator.utils.Database import DatabaseItem, Database
from phoneme_classificator.utils.ProjectData import ProjectData

# Filter audios
filter_by_len = True
max_audio_len = 50000
min_audio_len = 35000

# Plot audio characteristics
show_plots = True
audio_lengths = []
feature_lengths = []

# Configuration of the features
feature_config = FeatureConfig()
feature_config.feature_type = 'mffc'
feature_config.nfft = 1024
feature_config.winlen = 20
feature_config.winstride = 10
feature_config.preemph = 0.98
feature_config.num_filters = 48
feature_config.num_ceps = 26

# Load project data
project_data = ProjectData()

database = Database(project_data)

# Get the names of each wav file in the directory
wav_names = os.listdir(project_data.WAV_DIR)
for wav_index in range(len(wav_names)):

    # Get filenames
    wav_filename = project_data.WAV_DIR + '/' + wav_names[wav_index]
    label_filename = project_data.PHONEMES_DIR + '/' + wav_names[wav_index].split(".")[0] + '.PHN'

    # Create database item
    item = DatabaseItem.fromFile(
        wav_name=wav_filename,
        label_name=label_filename,
        feature_config=feature_config)

    audio_lengths.append(len(item.getFeature().getAudio()))
    feature_lengths.append(len(item.getFeature().getFeature()))

    if filter_by_len:
        audio_len = len(item.getFeature().getAudio())
        if audio_len > max_audio_len or audio_len < min_audio_len:
            print('Audio ({}) discarded by length filter'.format(wav_names[wav_index]))
            continue
    # Add the new data to the database
    database.append(item)

    percentage = wav_index / len(wav_names) * 100
    print('Completed ' + str(int(percentage)) + '%')

print("Database generated")
print("Number of elements in database: " + str(len(database)))

print('Maxmium audio length: ' + str(max(audio_lengths)))
print('Maxmium label length: ' + str(max(feature_lengths)))

# Save the database into a file
train_database, val_database, test_database = database.get_training_databases(0.9, 0.1, 0.0)
# train_database.save(project_data.TRAIN_DATABASE_FILE)
# val_database.save(project_data.VAL_DATABASE_FILE)
# test_database.save(project_data.TEST_DATABASE_FILE)
print("Databases saved")

if show_plots:
    fig = plt.figure()

    fig_1 = fig.add_subplot(211)
    # if filter_by_len:
    fig_1.plot(range(len(audio_lengths)), audio_lengths)
    if filter_by_len:
        fig_1.axhline(y=max_audio_len, color='r', linestyle='-', xmin=0, xmax=len(audio_lengths))
        fig_1.axhline(y=min_audio_len, color='r', linestyle='-', xmin=0, xmax=len(audio_lengths))
    fig_1.set_title('Audio lengths')

    fig_2 = fig.add_subplot(212)
    fig_2.plot(range(len(feature_lengths)), feature_lengths)
    fig_2.set_title('Feature lengths')

    plt.show()
