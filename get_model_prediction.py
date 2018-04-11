from keras.models import load_model
from pre_model import pickle_if_not_pickled, reshape_for_conv2d
import numpy as np
import random
random.seed(21)
#TODO: Figure out how to set random seed as global var for whole project (I believe it gets reset every time random is imported

def get_model_prediction(model, cqt_segment_reshaped):
    midi_prediction = model.predict(cqt_segment_reshaped)
    print(midi_prediction.shape)
    # in the midi_prediction as is, the nums in the array are not probabilities (exactly) more
    # like an indication of whether or not the value should be a 0 or 1 (ie .52 is probably a 1,
    # .01 is probably a 0)

def main():
    model = load_model('weights-improvement-09-0.2496.hdf5') #TODO: Make this an arg
    cqt_segments, midi_segments = pickle_if_not_pickled()
    #TODO: I def dont' need all these vars -> fig out best way to get just necessary ones
    cqt_test_array_reshaped, cqt_train_array_reshaped, cqt_valid_array_reshaped, input_height, input_width, \
    midi_test_array_reshaped, midi_train_array_reshaped, \
    midi_valid_array_reshaped, one_D_array_len = reshape_for_conv2d(cqt_segments, midi_segments)
    example_cqt_segment_reshaped = random.choice(cqt_train_array_reshaped)
    num_examples = 1
    input_depth = 1
    example_cqt_segment_reshaped_for_one = np.reshape(
        example_cqt_segment_reshaped, (num_examples, input_height, input_width, input_depth))
    get_model_prediction(model, example_cqt_segment_reshaped_for_one)

if __name__ == '__main__':
    main()