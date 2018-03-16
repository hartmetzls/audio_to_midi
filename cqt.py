from create_dataset import find_audio_files, load_audio, create_simplified_midi, load_segment_of_audio_and_save
import librosa
import ntpath
from mido import MidiFile
import numpy as np
import matplotlib.pyplot as plt

def audio_segments_cqt(audio_segment_time_series, sr):
    desired_cqt_length_minus_1 = 48
    minimum_factor_cqt_hop_length = 64
    x = int(audio_segment_time_series.shape[0]/(
            minimum_factor_cqt_hop_length*desired_cqt_length_minus_1))
    hop_length = minimum_factor_cqt_hop_length * x
    C = librosa.cqt(audio_segment_time_series, sr=sr, hop_length=hop_length)
    print("cqt:", C.shape)
    if C.shape[1] != 49:
        print("CQT incorrect length")
    return C
    #Desired CQT length 49

    # print(C)
    # print("Cqt real:", np.array(C, 'float32'))
    #We don't care about the imaginary part bc we don't care where in the wave we are when we
    #  start reading it
    # CQT = librosa.amplitude_to_db(C, ref=np.max)
    # stft = librosa.core.stft(time_series_and_sr[0])
    # print("stft:", stft.shape)
    # librosa.display.specshow(CQT, y_axis='cqt_note')
    # plt.colorbar(format='%+2.0f dB')

def main():
    # compare_beats("C:/Users/Lilly/audio_and_midi/audio/Bach_BWV871-02_002_20090916-SMD.wav")
    # compare_beats("C:/Users/Lilly/audio_and_midi/audio/Bartok_SZ080-02_002_20110315-SMD.wav")

    audio_segments_cqt()


if __name__ == '__main__':
    main()