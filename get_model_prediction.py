# from numpy.random import seed
# seed(21)
# from tensorflow import set_random_seed
# set_random_seed(21)
# import random
# random.seed(21)

from keras.models import load_model
from models import pickle_if_not_pickled, reshape_for_dense, split, root_mse, reshape_for_conv2d, r2_coeff_determination
from exploratory_visualization import visualize_cqt, visualize_midi
import numpy as np
import matplotlib.pyplot as plt
from normalisation import *

import librosa
import librosa.display
from sklearn.metrics import mean_squared_error
from math import sqrt

def get_model_pred(model, cqt_segment_reshaped, midi_true, midi_height, midi_width):
    midi_pred = model.predict(cqt_segment_reshaped)

    # regarding loss, where does this sample stand compared to the loss for the whole set
    remove_num_samples_dim = np.squeeze(midi_pred, axis=0)
    mse_loss = mean_squared_error(midi_true, remove_num_samples_dim)
    rmse = sqrt(mse_loss)
    print("rmse:")
    print(rmse)

    midi_pred_unflattened = np.reshape(midi_pred, (midi_height, midi_width))
    midi_pred_rounded = np.rint(midi_pred_unflattened)
    midi_pred_rounded_unflattened = np.reshape(midi_pred_rounded, (midi_height, midi_width))
    print("midi_pred_unflattened:")
    midi_true_unflattened = np.reshape(midi_true, (midi_height, midi_width))
    visualize_midi(midi_pred_unflattened, title='MIDI Prediction')

    print("midi_pred_rounded_unflattened:")
    visualize_midi(midi_pred_rounded_unflattened, title='MIDI Prediction Rounded')

    print("midi_true:")
    visualize_midi(midi_true_unflattened, title='MIDI True')

def main():
    filepath = "model_and_visualizations.1363/weights-improvement-38-0.1363.hdf5"
    # filepath = "model_checkpoints/previous_run_architectures/weights-improvement-37-0.1344.hdf5"
    model = load_model(filepath,
                       custom_objects={'root_mse': root_mse, 'r2_coeff_determination': r2_coeff_determination})
    cqt_segments, midi_segments = pickle_if_not_pickled()
    example_midi = midi_segments[0]
    midi_height, midi_width = example_midi.shape
    # cqt_segments_reshaped, midi_segments_reshaped = reshape_for_dense(cqt_segments, midi_segments)
    cqt_segments_reshaped, midi_segments_reshaped = reshape_for_conv2d(cqt_segments, midi_segments)
    cqt_train, cqt_valid, cqt_test, midi_train, midi_valid, midi_test = split(
        cqt_segments_reshaped, midi_segments_reshaped)

    # look at one validation example
    num_validation_samples = len(cqt_valid)
    random_index = np.random.randint(num_validation_samples)
    example_cqt_segment = cqt_valid[random_index]
    midi_true = midi_valid[random_index]
    num_examples = 1
    input_height, input_width, input_depth = example_cqt_segment.shape
    example_cqt_segment_reshaped = example_cqt_segment.reshape(num_examples, input_height, input_width, input_depth)
    # get_model_pred(model, example_cqt_segment_reshaped, midi_true, midi_height, midi_width)

    valid_score = model.evaluate(cqt_valid, midi_valid)
    print("valid score:")
    print("[loss (rmse), root_mse, mae, r2_coeff_determination]")
    print(valid_score)

    # final test score
    score = model.evaluate(cqt_test, midi_test)
    print("[loss (rmse), root_mse, mae, r2_coeff_determination]")
    print(score)

    # look at one test example, including the cqt
    num_test_samples = len(cqt_test)
    random_index_test = np.random.randint(num_test_samples)
    # index for the sample referenced in the Free-form Visualization section
    random_index_test = 1498

    print("random index test:")
    print(random_index_test)
    example_cqt_segment_test = cqt_test[random_index_test]

    # visualize cqt power spectrum (for one segment)
    depth_removed = np.squeeze(example_cqt_segment_test, axis=2)
    visualize_cqt(depth_removed)

    # alternate visualization: flip cqt to match midi heatmap y axis
    cqt_low_notes_at_top = np.flipud(depth_removed)
    visualize_cqt(cqt_low_notes_at_top, flipped=True)

    midi_true_test = midi_test[random_index_test]
    example_test_cqt_segment_reshaped = example_cqt_segment_test.reshape(
        num_examples, input_height, input_width, input_depth)
    get_model_pred(model, example_test_cqt_segment_reshaped, midi_true_test, midi_height, midi_width)





if __name__ == '__main__':
    main()