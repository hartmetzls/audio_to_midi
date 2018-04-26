from models import pickle_if_not_pickled
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

def main():
    cqt_segments, midi_segments = pickle_if_not_pickled()
    random_index = np.random.randint(len(cqt_segments))
    example_cqt_segment = cqt_segments[random_index]

    # visualize cqt power spectrum
    CQT = librosa.amplitude_to_db(example_cqt_segment, ref=np.max)
    librosa.display.specshow(CQT, x_axis='time', y_axis='cqt_note')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

    # visualize a cqt heatmap and it's corresponding midi heatmap
    plt.imshow(example_cqt_segment, cmap='hot', interpolation='nearest')
    plt.show()
    example_midi = midi_segments[random_index]
    plt.imshow(example_midi, cmap='hot', interpolation='nearest')
    plt.show()

if __name__ == '__main__':
    main()