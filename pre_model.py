from create_dataset import preprocess_audio_and_midi
import pickle
from sklearn.model_selection import train_test_split
import os
import numpy as np
import random

## env var to set GPU options#
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

def reshape_for_conv2d(cqt_segments, midi_segments):
    cqt_segments_array = np.array(cqt_segments)
    midi_segments_array = np.array(midi_segments)

    #a_few_examples vars created for quick testing
    cqt_segments_array = cqt_segments_array[:10]
    midi_segments_array = midi_segments_array[:10]

    #adds depth dimension to cqt segment (necessary for Conv2D)
    example_cqt_segment = cqt_segments_array[0]
    input_height, input_width = example_cqt_segment.shape
    however_many_there_are = -1
    cqt_segments_reshaped = cqt_segments_array.reshape(however_many_there_are, input_height, input_width, 1)

    #reshape output (necessary for Conv2D)
    example_midi_segment = midi_segments_array[0]
    output_height, output_width = example_midi_segment.shape
    one_D_array_len = output_height * output_width
    midi_segments_reshaped = midi_segments_array.reshape(however_many_there_are, one_D_array_len) #(-1,
    # one_D_array_len) does the same thing as num_segments, oneDaray len

    return cqt_segments_reshaped, midi_segments_reshaped

def split(cqt_segments_reshaped, midi_segments_reshaped):
    # shuffles before splitting by default
    cqt_train_and_valid, cqt_test, midi_train_and_valid, midi_test = train_test_split(
        cqt_segments_reshaped, midi_segments_reshaped, test_size=0.2, random_state=21)
    cqt_train, cqt_valid, midi_train, midi_valid = train_test_split(
        cqt_train_and_valid, midi_train_and_valid, test_size=0.2, random_state=21)
    return cqt_train, cqt_valid, cqt_test, midi_train, midi_valid, midi_test

def conv2d_model(cqt_train, cqt_valid, cqt_test, midi_train, midi_valid, midi_test):
    #BTW it is not expected that a CNN should be able to take a input of any shape (even if it
    # had the correct number of dims). These numbers (input height, weight, output len) could
    # reasonably be hard-coded vars.
    example_cqt_segment = cqt_train[0]
    input_height, input_width, input_depth = example_cqt_segment.shape
    example_midi_segment = midi_train[0]
    one_D_array_len = len(example_midi_segment)

    model = Sequential()
    #to try 24*1 kernel (column)
    model.add(Conv2D(filters=2, kernel_size=2, strides=1, padding='valid', activation='relu',
                     input_shape=(input_height, input_width, 1))) #valid doesn't go beyond input
    # edge - great for this prob bc we already have padding (most of the time)
    # re freq, however, same is more appropriate (or pad with zeros)

    #pooling is mostly for reducing dimensionality, but you don't need that so much (compared to
    # a classification problem)
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    #Make sure EVERY layer in the network has at least as many nodes as the output size (h * w)
    model.add(Dense(one_D_array_len, activation='sigmoid'))
    model.summary()
    model.compile(loss='mean_squared_error',
                  optimizer='sgd')
    epochs = 10
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpointer = ModelCheckpoint(filepath=filepath, monitor='loss',
                                   verbose=1, save_best_only=True, save_weights_only=False)
    model.fit(cqt_train, midi_train,
              validation_data=(cqt_valid, midi_valid),
              epochs=epochs, batch_size=1, callbacks=[checkpointer], verbose=1)
    score = model.evaluate(cqt_test, midi_test)
    print(score)

def depickle_and_model_architecture():
    random.seed(21)
    np.random.seed(21)
    cqt_segments, midi_segments = pickle_if_not_pickled()
    cqt_segments_reshaped, midi_segments_reshaped = reshape_for_conv2d(cqt_segments, midi_segments)
    cqt_train, cqt_valid, cqt_test, midi_train, midi_valid, midi_test = split(
        cqt_segments_reshaped, midi_segments_reshaped)
    conv2d_model(cqt_train, cqt_valid, cqt_test, midi_train, midi_valid, midi_test)

def main():
    depickle_and_model_architecture()

if __name__ == '__main__':
    main()