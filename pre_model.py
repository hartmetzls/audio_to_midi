from create_dataset import preprocess_audio_and_midi
import pickle
from sklearn.model_selection import train_test_split

def pickle_if_not_pickled():
    try:
        with open('cqt_segments_midi_segments.pkl', 'rb') as handle:
            # pickle.load(handle)
            # cqt_segments, midi_segments = pickle.load(handle)
        # this instead if above code is buggy
    except (OSError, IOError) as handle:
        preprocess_audio_and_midi()
        pickle_if_not_pickled() #TODO: Ask lou if this is an acceptable way for this function to be set up
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