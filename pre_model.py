from create_dataset import preprocess_audio_and_midi
import pickle
from sklearn.model_selection import train_test_split
import os

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

def create_feature_sets_and_labels(cqt_segments, midi_segments): #TODO: Validation set as well?
    cqt_train, cqt_test, midi_train, midi_test = train_test_split(cqt_segments, midi_segments, test_size=0.25, random_state=21) #shuffles before splitting
    print("butts")

def pre_model_architecture():
    cqt_segments, midi_segments = pickle_if_not_pickled()
    create_feature_sets_and_labels(cqt_segments, midi_segments)

def main():
    pre_model_architecture()

if __name__ == '__main__':
    main()