from create_dataset import preprocess_audio_and_midi
import pickle
from sklearn.model_selection import train_test_split
import os
import numpy as np
import random

## extra imports to set GPU options
# #TODO: ASK Lou why this fix works? (For error ...GetConvolveAlgorithms... (see Capstone google doc))
# ie I only have one GPU...how was it not visible to tf as device 0?
#
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Includes layer options to be tried later
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

def pickle_if_not_pickled():
    try:
        with open('cqt_segments_midi_segments.pkl', 'rb') as handle:
            cqt_segments = pickle.load(handle)
            midi_segments = pickle.load(handle)
    except (OSError, IOError) as err:
        # Windows
        if os.name == 'nt':
            directory_str = "C:/Users/Lilly/audio_and_midi/"
        # Linux
        if os.name == 'posix':
            directory_str = "/home/lilly/Downloads/audio_midi/"
        preprocess_audio_and_midi(directory_str)
        cqt_segments, midi_segments = pickle_if_not_pickled()
    return cqt_segments, midi_segments

def split_and_reshape_for_conv2d(cqt_segments, midi_segments):

    #a_few_examples vars created for quick testing
    cqt_segments_a_few_examples = cqt_segments[:40]
    midi_segments_a_few_examples = midi_segments[:40]

    # shuffles before splitting by default
    cqt_train_and_valid, cqt_test, midi_train_and_valid, midi_test = train_test_split(
        cqt_segments_a_few_examples, midi_segments_a_few_examples, test_size=0.2, random_state=21)
    cqt_train, cqt_valid, midi_train, midi_valid = train_test_split(
        cqt_train_and_valid, midi_train_and_valid, test_size=0.2, random_state=21)

    #TODO: Look into diff btwn array and asarray and confirm that you're using both correctly in this project
    cqt_train_array = np.array(cqt_train)
    midi_train_array = np.array(midi_train)
    cqt_valid_array = np.array(cqt_valid)
    midi_valid_array = np.array(midi_valid)
    cqt_test_array = np.array(cqt_test)
    midi_test_array = np.array(midi_test)

    #makes output 1D (This is potentially only a hold-over until I find out how to have 2D output with a CNN)
    midi_train_array_flattened = np.ndarray.flatten(midi_train_array)
    midi_valid_array_flattened = np.ndarray.flatten(midi_valid_array)
    midi_test_array_flattened = np.ndarray.flatten(midi_test_array)

    example_cqt_segment = cqt_segments[0]
    input_height, input_width = example_cqt_segment.shape

    #for debugging
    cqt_train_array_shape = cqt_train_array.shape
    example_midi = midi_segments[0]

    output_height, output_width = example_midi.shape
    num_training_cqts = len(cqt_train_array)
    num_validation_cqts = len(cqt_valid_array)
    num_testing_cqts = len(cqt_test_array)

    #adds depth dimension to cqt segment (necessary for Conv2D i believe?)
    cqt_train_array_reshaped = cqt_train_array.reshape(num_training_cqts, input_height, input_width, 1)
    cqt_valid_array_reshaped = cqt_valid_array.reshape(num_validation_cqts, input_height, input_width, 1)
    cqt_test_array_reshaped = cqt_test_array.reshape(num_testing_cqts, input_height, input_width, 1)

    #reshape (necessary for Conv2D i believe?)
    one_D_array_len = output_height * output_width
    midi_train_array_flattened_reshaped = midi_train_array_flattened.reshape(num_training_cqts, one_D_array_len)
    midi_valid_array_flattened_reshaped = midi_valid_array_flattened.reshape(num_validation_cqts, one_D_array_len)
    midi_test_array_flattened_reshaped = midi_test_array_flattened.reshape(num_testing_cqts, one_D_array_len)

    return cqt_test_array_reshaped, cqt_train_array_reshaped, cqt_valid_array_reshaped, input_height, input_width, \
           midi_test_array_flattened_reshaped, midi_train_array_flattened_reshaped, \
           midi_valid_array_flattened_reshaped, one_D_array_len

def conv2d_model(cqt_test_array_reshaped, cqt_train_array_reshaped, cqt_valid_array_reshaped, input_height, input_width,
                 midi_test_array_flattened_reshaped, midi_train_array_flattened_reshaped,
                 midi_valid_array_flattened_reshaped, one_D_array_len):
    model = Sequential()
    model.add(Conv2D(filters=2, kernel_size=2, strides=1, padding='valid', activation='relu',
                     input_shape=(input_height, input_width, 1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(one_D_array_len, activation='softmax'))
    model.summary()
    model.compile(loss='mean_squared_error',
                  optimizer='sgd')
    epochs = 10
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpointer = ModelCheckpoint(filepath=filepath, monitor='loss',
                                   verbose=1, save_best_only=True, save_weights_only=False)
    model.fit(cqt_train_array_reshaped, midi_train_array_flattened_reshaped,
              validation_data=(cqt_valid_array_reshaped, midi_valid_array_flattened_reshaped),
              epochs=epochs, batch_size=1, callbacks=[checkpointer], verbose=1)
    score = model.evaluate(cqt_test_array_reshaped, midi_test_array_flattened_reshaped)
    print(score)

def depickle_and_model_architecture():
    random.seed(21)
    np.random.seed(21)
    cqt_segments, midi_segments = pickle_if_not_pickled()
    cqt_test_array_reshaped, cqt_train_array_reshaped, cqt_valid_array_reshaped, input_height, input_width,\
    midi_test_array_flattened_reshaped, midi_train_array_flattened_reshaped, \
    midi_valid_array_flattened_reshaped, one_D_array_len = split_and_reshape_for_conv2d(cqt_segments, midi_segments)
    conv2d_model(cqt_test_array_reshaped, cqt_train_array_reshaped, cqt_valid_array_reshaped, input_height, input_width,
                 midi_test_array_flattened_reshaped, midi_train_array_flattened_reshaped,
                 midi_valid_array_flattened_reshaped, one_D_array_len)

def main():
    depickle_and_model_architecture()

if __name__ == '__main__':
    main()