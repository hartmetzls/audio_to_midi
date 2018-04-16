import random
import pickle
from sklearn.metrics import mean_squared_error
import numpy as np
from models import reshape_for_conv2d
from math import sqrt

def benchmark():
    with open('cqt_segments_midi_segments.pkl', 'rb') as handle:
        cqt_segments = pickle.load(handle)
        midi_segments = pickle.load(handle)

    cqt_segments_reshaped, midi_segments_reshaped = reshape_for_conv2d(cqt_segments, midi_segments)
    midi_segments_reshaped_copy = np.copy(midi_segments_reshaped)
    np.random.shuffle(midi_segments_reshaped_copy)

    mse_loss = mean_squared_error(midi_segments_reshaped, midi_segments_reshaped_copy)
    print(mse_loss)
    rmse = sqrt(mse_loss)
    print(rmse)

def main():
    random.seed(21)
    np.random.seed(21)
    benchmark()

if __name__ == '__main__':
    main()