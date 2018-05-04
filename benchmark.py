import random
random.seed(21)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
np.random.seed(21)
from models import reshape_for_conv2d
from math import sqrt
from models import pickle_if_not_pickled

def benchmark():
    cqt_segments, midi_segments = pickle_if_not_pickled()

    # flatten midi
    cqt_segments_reshaped, midi_segments_reshaped = reshape_for_conv2d(cqt_segments, midi_segments)
    midi_segments_reshaped_benchmark_pred = np.array(midi_segments_reshaped)
    np.random.shuffle(midi_segments_reshaped_benchmark_pred)

    mse_loss = mean_squared_error(midi_segments_reshaped, midi_segments_reshaped_benchmark_pred)
    rmse = sqrt(mse_loss)
    print("rmse:")
    print(rmse)

    mae = mean_absolute_error(midi_segments_reshaped, midi_segments_reshaped_benchmark_pred)
    print("mae:")
    print(mae)

    r2 = r2_score(midi_segments_reshaped, midi_segments_reshaped_benchmark_pred)
    print("r2:")
    print(r2)

def main():
    benchmark()

if __name__ == '__main__':
    main()