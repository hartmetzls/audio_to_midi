import random
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from models import reshape_for_conv2d
from math import sqrt
from models import pickle_if_not_pickled

def benchmark():
    cqt_segments, midi_segments = pickle_if_not_pickled()

    # use existing function to flatten midi for sklearn function
    cqt_segments_reshaped, midi_segments_reshaped = reshape_for_conv2d(cqt_segments, midi_segments)
    midi_segments_reshaped_copy = np.array(midi_segments_reshaped)
    np.random.shuffle(midi_segments_reshaped_copy)

    mse_loss = mean_squared_error(midi_segments_reshaped, midi_segments_reshaped_copy)
    rmse = sqrt(mse_loss)
    print("rmse:")
    print(rmse)

    mae = mean_absolute_error(midi_segments_reshaped, midi_segments_reshaped_copy)
    print("mae:")
    print(mae)

    r2 = r2_score(midi_segments_reshaped, midi_segments_reshaped_copy)
    print("r2:")
    print(r2)

def main():
    random.seed(21)
    np.random.seed(21)
    benchmark()

if __name__ == '__main__':
    main()