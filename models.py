# from numpy.random import seed
# seed(21)
# from tensorflow import set_random_seed
# set_random_seed(21)
# import random
# random.seed(21)

from create_dataset import preprocess_audio_and_midi, done_beep
import pickle
from sklearn.model_selection import train_test_split
import os
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import time

# env var to set GPU options
# (this was necessary for my machine. comment out line below if it throws an error.)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from keras.layers import Conv2D
from keras.layers import Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import optimizers
from keras import backend as K

# if the data is not already in a pkl file in the project directory,
# save cqt_segments_midi_segments.pkl to project directory
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
    # convert data to np array in order to pass data to keras functions
    cqt_segments_array = np.array(cqt_segments)
    midi_segments_array = np.array(midi_segments)

    # this is a convenient place to choose to run a portion of the dataset (for quick testing)
    cqt_segments_array = cqt_segments_array[:]
    midi_segments_array = midi_segments_array[:]

    # adds depth dimension to cqt segment (necessary for Conv2D)
    example_cqt_segment = cqt_segments_array[0]
    input_height, input_width = example_cqt_segment.shape
    however_many_there_are = -1
    cqt_segments_reshaped = cqt_segments_array.reshape(however_many_there_are, input_height, input_width, 1)

    # reshape output for Flatten layer
    example_midi_segment = midi_segments_array[0]
    output_height, output_width = example_midi_segment.shape
    one_d_array_len = output_height * output_width
    midi_segments_reshaped = midi_segments_array.reshape(however_many_there_are, one_d_array_len)

    return cqt_segments_reshaped, midi_segments_reshaped

def reshape_for_dense(cqt_segments, midi_segments):
    cqt_segments_array = np.array(cqt_segments)
    midi_segments_array = np.array(midi_segments)

    # this is a convenient place to choose to run a portion of the dataset (for quick testing)
    cqt_segments_array = cqt_segments_array[:]
    midi_segments_array = midi_segments_array[:]

    # debugging nan loss (referenced in Implementation section)
    # check_cqt_infs = np.where(np.isinf(cqt_segments_array))
    # check_midi_infs = np.where(np.isinf(midi_segments_array))
    # print(check_cqt_infs)
    # print(check_midi_infs)
    # check_cqt_nans = np.where(np.isnan(cqt_segments_array))
    # check_midi_nans = np.where(np.isnan(midi_segments_array))
    # print(check_cqt_nans)
    # print(check_midi_nans)

    example_cqt_segment = cqt_segments_array[0]
    input_height, input_width = example_cqt_segment.shape

    however_many_there_are = -1

    # reshape output for Flatten layer
    example_midi_segment = midi_segments_array[0]
    output_height, output_width = example_midi_segment.shape
    one_d_array_len = output_height * output_width
    midi_segments_reshaped = midi_segments_array.reshape(however_many_there_are, one_d_array_len)

    return cqt_segments_array, midi_segments_reshaped

def split(cqt_segments_reshaped, midi_segments_reshaped):
    # shuffles before splitting by default
    cqt_train_and_valid, cqt_test, midi_train_and_valid, midi_test = train_test_split(
        cqt_segments_reshaped, midi_segments_reshaped, test_size=0.2, random_state=21)

    cqt_train, cqt_valid, midi_train, midi_valid = train_test_split(
        cqt_train_and_valid, midi_train_and_valid, test_size=0.2, random_state=21)
    return cqt_train, cqt_valid, cqt_test, midi_train, midi_valid, midi_test

def conv2d_model(cqt_train, cqt_valid, cqt_test, midi_train, midi_valid, midi_test):
    # this is a convenient point to confirm whether or not the full dataset is being run
    print("num training examples:")
    print(len(cqt_train))

    example_cqt_segment = cqt_train[0]
    input_height, input_width, input_depth = example_cqt_segment.shape
    example_midi_segment = midi_train[0]
    one_d_array_len = len(example_midi_segment)

    model = Sequential()
    model.add(Conv2D(filters=2, kernel_size=(1, 2), strides=(1), padding='same', activation='relu',
                         input_shape=(input_height, input_width, 1)))
    model.add(Conv2D(filters=2, kernel_size=(7, 1), strides=(1), padding='same', activation='relu'))
    model.add(Conv2D(filters=3, kernel_size=(1, 2), strides=(1), padding='same', activation='relu'))
    model.add(Conv2D(filters=3, kernel_size=(7, 1), strides=(1), padding='same', activation='relu'))
    for i in range(2):
        model.add(Conv2D(filters=4, kernel_size=(1, 2), strides=(1, 2), padding='same', activation='relu'))
    for i in range(3):
        model.add(Conv2D(filters=5, kernel_size=(1, 2), strides=(1, 2), padding='same', activation='relu'))
    model.add(Conv2D(filters=6, kernel_size=(1, 2), strides=(1), padding='same', activation='relu'))

    model.add(Flatten())
    model.add(Dense(one_d_array_len, activation='sigmoid'))
    model.summary()
    adam = optimizers.adam(lr=0.0001, decay=.00001)
    model.compile(loss=root_mse,
                  optimizer=adam,
                  metrics=[root_mse, 'mae', r2_coeff_determination])
    epochs = 100
    filepath = "model_checkpoints/weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
    checkpointer = ModelCheckpoint(filepath=filepath, monitor='val_loss',
                                   verbose=1, save_best_only=True, save_weights_only=False)

    # create a callback tensorboard object:
    tensorboard = TensorBoard(log_dir='./tensorboard_logs', histogram_freq=0, batch_size=1, write_graph=True,
                                write_grads=True, write_images=True, embeddings_freq=0,
                                embeddings_layer_names=None, embeddings_metadata=None)

    history_for_plotting = model.fit(cqt_train, midi_train,
              validation_data=(cqt_valid, midi_valid),
              epochs=epochs, batch_size=32, callbacks=[checkpointer, tensorboard], verbose=1)
    score = model.evaluate(cqt_test, midi_test)

    # completely optional. plays a sound when the model finishes running
    done_beep()

    # also optional. times the runtime (thus far) and shows the time per epoch
    total_time = time.time() - start_time
    print("--- %s seconds ---" % (total_time))
    print("each epoch:")
    print(total_time / epochs)

    # test run only
    print("test run score:")
    print("[loss (rmse), root_mse, mae, r2_coeff_determination]")
    print(score)

    #summarize history for loss
    #https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
    plt.plot(history_for_plotting.history['loss'])
    plt.plot(history_for_plotting.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train rmse', 'validation rmse'], loc='upper right')
    plt.show()

    plt.plot(history_for_plotting.history['r2_coeff_determination'])
    plt.title('r2')
    plt.ylabel('r2_coeff_determination')
    plt.xlabel('epoch')
    plt.legend(['r2'], loc='upper left')
    plt.show()

    plt.plot(history_for_plotting.history['mean_absolute_error'])
    plt.title('mae')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['mae'], loc='upper right')
    plt.show()

def dense_model(cqt_train, cqt_valid, cqt_test, midi_train, midi_valid, midi_test):
    example_cqt_segment = cqt_train[0]
    input_height, input_width = example_cqt_segment.shape
    example_midi_segment = midi_train[0]
    one_D_array_len = len(example_midi_segment)
    model = Sequential()
    model.add(Dense(1044, input_shape=(input_height, input_width), activation='relu'))
    model.add(Flatten())
    model.add(Dense(one_D_array_len, activation='sigmoid'))
    model.summary()
    model.compile(loss=root_mse,
                  optimizer='adam')
    epochs = 100
    filepath = "model_checkpoints/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpointer = ModelCheckpoint(filepath=filepath, monitor='loss',
                                   verbose=1, save_best_only=True, save_weights_only=False)
    history_for_plotting = model.fit(cqt_train, midi_train,
                                     validation_data=(cqt_valid, midi_valid),
                                     epochs=epochs, batch_size=1, callbacks=[checkpointer], verbose=1)
    score = model.evaluate(cqt_test, midi_test)

    # summarize history for loss
    # https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
    plt.plot(history_for_plotting.history['loss'])
    plt.plot(history_for_plotting.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def root_mse(y_true, y_pred):
    # returns tensorflow.python.framework.ops.Tensor
    return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred))))

# https://jmlb.github.io/ml/2017/03/20/CoeffDetermination_CustomMetric4Keras/
def r2_coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    # epsilon avoids division by zero
    return (1 - SS_res / (SS_tot + K.epsilon()))

def depickle_and_model_architecture():
    cqt_segments, midi_segments = pickle_if_not_pickled()
    cqt_segments_reshaped, midi_segments_reshaped = reshape_for_conv2d(cqt_segments, midi_segments)
    # cqt_segments_reshaped, midi_segments_reshaped = reshape_for_dense(cqt_segments, midi_segments)
    cqt_train, cqt_valid, cqt_test, midi_train, midi_valid, midi_test = split(
        cqt_segments_reshaped, midi_segments_reshaped)
    conv2d_model(cqt_train, cqt_valid, cqt_test, midi_train, midi_valid, midi_test)
    # dense_model(cqt_train, cqt_valid, cqt_test, midi_train, midi_valid,
    #             midi_test)

def main():
    depickle_and_model_architecture()

if __name__ == '__main__':
    # set start time here in order to clock runtime (incl. time per epoch) before metrics plots show
    start_time = time.time()
    main()