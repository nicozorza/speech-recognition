import os
import tensorflow as tf
from PIL import Image

from phoneme_classificator.utils.Database import DatabaseItem, Database
from phoneme_classificator.utils.Label import Phoneme
from phoneme_classificator.utils.ProjectData import ProjectData
import numpy as np


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
    label_filename = project_data.PHONEMES_DIR + '/' + wav_names[wav_index].split(".")[0] + '.PHN'

    # Create database item
    item = DatabaseItem.fromFile(
        wav_name=wav_filename,
        label_name=label_filename,
        nfft=project_data.fft_points,
        window_len=project_data.frame_length,
        win_stride=project_data.frame_stride)

    # Add the new data to the database
    database.append(item)

print("Database generated")
print("Number of elements in database: " + str(len(database)))


# Save the database into a file
database.save(project_data.DATABASE_FILE)
print("Database saved in:", project_data.DATABASE_NAME)

# record_iterator = tf.python_io.tf_record_iterator(path=project_data.DATABASE_FILE)
#
# for string_record in record_iterator:
#     example = tf.train.Example()
#     example.ParseFromString(string_record)
#
#     height = int(example.features.feature['seq_len'].int64_list.value[0])
#
#     width = int(example.features.feature['nfft'].int64_list.value[0])
#
#     img_string = example.features.feature['feature'].float_list.value
#
#     annotation_string = (example.features.feature['label'].int64_list.value)
#
#     img_1d = np.asarray(img_string, dtype=np.float32)
#     reconstructed_img = img_1d.reshape((height, width))
#     aux1=database.getItemFromIndex(0).getFeature().getFeature()[0]
#     aux2=reconstructed_img[0]
#
#
#     annotation_1d = np.asarray(annotation_string, dtype=np.int64)
#
#     print("sd")
#
#
#
#
