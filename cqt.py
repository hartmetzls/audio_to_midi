from create_dataset import find_audio_files, load_audio, create_simplified_midi
import librosa
import ntpath
from mido import MidiFile
import numpy as np
import matplotlib.pyplot as plt

def explore_stft():
    directory_str = "C:/Users/Lilly/audio_and_midi/four_beats/audio"
    audio_clips_no_duplicates = find_audio_files(directory_str)
    for audio_clip in audio_clips_no_duplicates[:5]:
        print("audio clip:", audio_clip)
        time_series_and_sr = load_audio(audio_clip)
        # print("time series shape:", time_series_and_sr[0].shape)
        desired_cqt_length_minus_1 = 48
        minimum_factor_cqt_hop_length = 64
        x = int(time_series_and_sr[0].shape[0]/(
                minimum_factor_cqt_hop_length*desired_cqt_length_minus_1))
        hop_length = minimum_factor_cqt_hop_length * x
        C = librosa.cqt(time_series_and_sr[0], sr=time_series_and_sr[1], hop_length=hop_length)
        print("cqt:", C.shape)
        if C.shape[1] != 49:
            print("CQT incorrect length")
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

        midi_str = "C:/Users/Lilly/audio_and_midi/four_beats/midi/" + ntpath.basename(audio_clip)[
                                                                    :-4]\
                   + \
                   ".mid"
        midi_file = MidiFile(midi_str)
        dumbed_down_midi = create_simplified_midi(midi_file)
        print("ticks since start:", (dumbed_down_midi[1]))

def main():
    # compare_beats("C:/Users/Lilly/audio_and_midi/audio/Bach_BWV871-02_002_20090916-SMD.wav")
    # compare_beats("C:/Users/Lilly/audio_and_midi/audio/Bartok_SZ080-02_002_20110315-SMD.wav")

    explore_stft()


if __name__ == '__main__':
    main()