import librosa
import numpy as np

def audio_segments_cqt(audio_segment_time_series, sr):
    cqt_of_segment = librosa.cqt(audio_segment_time_series, sr=sr)
    # Remove the imaginary number part of the CQT (ie We don't care where in the wave we are when we start reading it)
    cqt_of_segment_real = cqt_of_segment.real
    cqt_of_segment_copy_real = np.array(cqt_of_segment_real, dtype='float32')
    return cqt_of_segment_copy_real

def main():
    audio_segments_cqt()

if __name__ == '__main__':
    main()