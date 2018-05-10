from models import pickle_if_not_pickled
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

def visualize_cqt(example_cqt_segment, flipped=False):
    # visualize cqt power spectrum (for one segment)
    # librosa.display.specshow(cqt_low_notes_at_top, sr=22050 * 4, x_axis='time', y_axis='cqt_note')

    # alternate visualization
    CQT = librosa.amplitude_to_db(example_cqt_segment, ref=np.max)
    if not flipped:
        librosa.display.specshow(CQT, sr=22050 * 4, x_axis='time', y_axis='cqt_note')
    else:
        librosa.display.specshow(CQT, sr=22050 * 4, x_axis='time')
        plt.ylabel("High frequencies (pitches) ---> Low frequencies (pitches)")
    plt.colorbar(format='%+2.0f dB')
    plt.title("CQT Power Spectrum")
    plt.show()

    if not flipped:
        # visualize a cqt heatmap and it's corresponding midi heatmap (for one segment)
        plt.imshow(example_cqt_segment, cmap='hot', interpolation='nearest')
        plt.title("CQT Heatmap")
        plt.xlabel("Time")
        plt.ylabel("Frequency bins")
        plt.show()

def visualize_midi(example_midi, title=None):
    plt.imshow(example_midi, cmap='hot', interpolation='nearest')
    if title == None:
        plt.title("MIDI Heatmap")
    else:
        plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("MIDI pitch number (scaled)")
    plt.show()

def main():
    cqt_segments, midi_segments = pickle_if_not_pickled()
    random_index = np.random.randint(len(cqt_segments))
    print("random index:")
    print(random_index)

    # index for sample in Exploratory Visualization section
    random_index = 8059

    example_cqt_segment = cqt_segments[random_index]

    # # flip cqt to match midi heatmap y axis
    # cqt_low_notes_at_top = np.flipud(example_cqt_segment)
    visualize_cqt(example_cqt_segment)

    example_midi = midi_segments[random_index]
    visualize_midi(example_midi)




if __name__ == '__main__':
    main()