import random
import pickle
from sklearn.metrics import mean_squared_error
import numpy as np

def benchmark():
    with open('cqt_segments_midi_segments.pkl', 'rb') as handle:
        cqt_segments = pickle.load(handle)
        midi_segments = pickle.load(handle)
    results = []
    for cqt_segment in cqt_segments:
        random_midi_segment = random.choice(midi_segments)
        results.append(random_midi_segment)
    midi_segments_array = np.asarray(midi_segments)
    midi_predicted_array = np.asarray(results)
    midi_segments_array_flattened = np.ndarray.flatten(midi_segments_array)
    midi_predicted_array_flattened = np.ndarray.flatten(midi_predicted_array)

    # debugging
    midi_segments_shape = midi_segments_array.shape
    midi_predicted_shape = midi_predicted_array.shape
    midi_segments_array_flattened_shape = midi_segments_array_flattened.shape
    midi_predicted_array_flattened_shape = midi_predicted_array_flattened.shape

    mse_regression_loss = mean_squared_error(midi_segments_array_flattened, midi_predicted_array_flattened)
    print(mse_regression_loss)

# TODO: Random seed reg, np,

def main():
    random.seed(21)
    benchmark()

if __name__ == '__main__':
    main()