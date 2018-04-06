from create_dataset import preprocess_audio_and_midi
import pickle
from sklearn.model_selection import train_test_split
import os
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

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

def split_and_model_architecture(cqt_segments, midi_segments): #TODO: Validation set as well?
    cqt_train, cqt_test, midi_train, midi_test = train_test_split(cqt_segments, midi_segments, test_size=0.25, random_state=21) #shuffles before splitting
    example_cqt_segment = cqt_segments[0]
    input_height, input_width = example_cqt_segment.shape
    num_training_cqts = len(cqt_train)
    model = Sequential()

    # Define your architecture.
    model.add(Conv2D(filters=16, kernel_size=2, padding='valid', activation='relu',
                     input_shape=(input_height, input_width, None))) #173, 84
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32, kernel_size=2, padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=2, padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(GlobalAveragePooling2D(data_format='channels_last'))
    model.add(Dense(133, activation='softmax'))
    model.summary()

    model.compile(loss='mean_squared_error',
              optimizer='sgd', metrics=['acc'])

    from keras.callbacks import ModelCheckpoint
    # specify the number of epochs that you would like to use to train the model.
    epochs = 100
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5',
                                   verbose=1, save_best_only=True)

    model.fit(cqt_train, midi_train,
              epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)

def depickle_and_model_architecture():
    cqt_segments, midi_segments = pickle_if_not_pickled()
    split_and_model_architecture(cqt_segments, midi_segments)

def main():
    depickle_and_model_architecture()

if __name__ == '__main__':
    main()