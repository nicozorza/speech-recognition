import os

from phoneme_classificator.utils.Database import DatabaseItem, Database
from phoneme_classificator.utils.ProjectData import ProjectData



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
    item = DatabaseItem.fromFile(wav_name=wav_filename, label_name=label_filename, window_len=20, win_stride=10)

    # Add the new data to the database
    database.append(item)
print("hola")

# Save the database into a file
database.save(project_data.DATABASE_FILE)

print("Database generated")
print("Number of elements in database: " + str(len(database)))
