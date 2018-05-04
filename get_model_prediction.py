from keras.models import load_model
from models import pickle_if_not_pickled, reshape_for_dense, split, root_mse, reshape_for_conv2d, r2_coeff_determination
import numpy as np
import matplotlib.pyplot as plt
from normalisation import *
import random
random.seed(21)
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
    plt.imshow(midi_pred_unflattened, cmap='hot', interpolation='nearest')
    plt.show()
    print("midi_pred_rounded_unflattened:")
    plt.imshow(midi_pred_rounded_unflattened, cmap='hot', interpolation='nearest')
    plt.show()
    print("midi_true:")
    plt.imshow(midi_true_unflattened, cmap='hot', interpolation='nearest')
    plt.show()

def main():
    filepath = "model_and_visualizations.1363/weights-improvement-38-0.1363.hdf5"
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

    # final test score
    score = model.evaluate(cqt_test, midi_test)
    print("[loss (rmse), root_mse, mae, r2_coeff_determination]")
    print(score)

    # look at one test example, including the cqt
    num_test_samples = len(cqt_test)
    random_index_test = np.random.randint(num_test_samples)

    # index for the sample referenced in the Justification section
    # random_index_test = 1498

    print("random index test:")
    print(random_index_test)
    example_cqt_segment_test = cqt_test[random_index_test]

    # visualize cqt power spectrum (for one segment)
    depth_removed = np.squeeze(example_cqt_segment_test, axis=2)
    CQT = librosa.amplitude_to_db(depth_removed, ref=np.max)
    librosa.display.specshow(CQT, sr=22050*4, x_axis='time', y_axis='cqt_note')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

    # alternate visualization: flip cqt to match midi heatmap y axis
    cqt_low_notes_at_top = np.flipud(depth_removed)
    CQT = librosa.amplitude_to_db(cqt_low_notes_at_top, ref=np.max)
    librosa.display.specshow(CQT, sr=22050 * 4, x_axis='time', y_axis=None)
    plt.colorbar(format='%+2.0f dB')
    plt.show()

    midi_true_test = midi_test[random_index_test]
    example_test_cqt_segment_reshaped = example_cqt_segment_test.reshape(
        num_examples, input_height, input_width, input_depth)
    get_model_pred(model, example_test_cqt_segment_reshaped, midi_true_test, midi_height, midi_width)





if __name__ == '__main__':
    main()