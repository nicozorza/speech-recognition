import os

from phoneme_classificator.utils.AudioFeature import FeatureConfig
from phoneme_classificator.utils.Database import DatabaseItem, Database
from phoneme_classificator.utils.ProjectData import ProjectData

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

max_length = 90010
aux_len = 0
aux_label_len = 0
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
    if len(item.getFeature().getAudio()) >= aux_len:
        aux_len = len(item.getFeature().getAudio())
    if len(item.getLabel().getPhonemes()) >= aux_label_len:
        aux_label_len = len(item.getLabel().getPhonemes())

    # Add the new data to the database
    database.append(item)

    percentage = wav_index / len(wav_names) * 100
    print('Completed ' + str(int(percentage)) + '%')

print("Database generated")
print("Number of elements in database: " + str(len(database)))

print('Maxmium audio length: ' + str(aux_len))
print('Maxmium label length: ' + str(aux_label_len))

# Save the database into a file
train_database, val_database, test_database = database.get_training_databases(0.9, 0.1, 0.0)
train_database.save(project_data.TRAIN_DATABASE_FILE)
val_database.save(project_data.VAL_DATABASE_FILE)
test_database.save(project_data.TEST_DATABASE_FILE)
print("Databases saved")

# batches=database.get_batches_list(11)
#
# database2 = Database.fromFile(project_data.DATABASE_FILE, project_data)
# asd = 1

# database2 = Database.fromFile(project_data.DATABASE_FILE, project_data)
#
# import tensorflow as tf
#
# def parse_test(example):
#     context_features = {
#         "seq_len": tf.FixedLenFeature([], dtype=tf.int64),
#         "nfft": tf.FixedLenFeature([], dtype=tf.int64)
#     }
#     sequence_features = {
#         "feature": tf.VarLenFeature(dtype=tf.float32),
#         "label": tf.VarLenFeature(dtype=tf.int64)
#     }
#
#     context_parsed, sequence_parsed = tf.parse_single_sequence_example(
#         serialized=example,
#         context_features=context_features,
#         sequence_features=sequence_features
#     )
#
#     return sequence_parsed, context_parsed

# data = tf.data.TFRecordDataset(project_data.DATABASE_FILE)
# data = data.map(parse_test)
# data = data.repeat().shuffle(buffer_size=3).batch(3)
#
# iterator = data.make_one_shot_iterator()
# feature_dict = iterator.get_next()
# with tf.Session() as sess:
#
#     sequence, context = sess.run(feature_dict)
#     print([context])
#     aa=sequence["label"]
#     bb=context["seq_len"]
#     asd = 2
#
#     X = tf.placeholder(dtype=tf.float32, shape=[None, None])
#     seq_length = tf.placeholder(tf.int32, [None])
#
#     basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=100)
#     outputs, states = tf.nn.dynamic_rnn(basic_cell, X, sequence_length=seq_length, dtype=tf.float32)
#
#     sess.run(tf.global_variables_initializer())
#     h_t, h_final = tf.nn.dynamic_rnn(basic_cell, X)  # all h:s and the final state

