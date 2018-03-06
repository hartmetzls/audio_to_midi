import pytest
from pytest import approx

from create_dataset import *

def test_dumbed_down_midi():
    directory_str_audio = "C:/Users/Lilly/audio_and_midi/audio"
    audio_files_no_duplicates = find_audio_files(directory_str_audio)
    for audio_file in audio_files_no_duplicates:
        # time_series_and_sr = load_audio(audio_file)
        # every_fourth_timestamp_array = timestamps_and_call_load_segment(audio_file, time_series_and_sr)
        midi_file = load_midi(audio_file)
        dumbed_down_track, _ = create_simplified_midi(midi_file)
        previous_message_time = 0
        for message in dumbed_down_track:
            cur_message_time = message[2]
            assert previous_message_time <= cur_message_time, "message out of order"
            previous_message_time = cur_message_time

        # divide_points = midi_divide_points(midi_file)

def test_every_fourth_timestamp_array():
    directory_str_audio = "C:/Users/Lilly/audio_and_midi/audio"
    audio_files_no_duplicates = find_audio_files(directory_str_audio)
    for audio_file in audio_files_no_duplicates:
        time_series_and_sr = load_audio(audio_file)
        every_fourth_timestamp_array = timestamps_and_call_load_segment(audio_file, time_series_and_sr)
        previous_timestamp = 0
        for time in every_fourth_timestamp_array:
            assert previous_timestamp <= time, "time out of order (time out of mind)"
            previous_timestamp = time

def test_duration():
    directory_str_audio = "C:/Users/Lilly/audio_and_midi/audio"
    audio_files_no_duplicates = find_audio_files(directory_str_audio)
    for audio_file in audio_files_no_duplicates:
        time_series_and_sr = load_audio(audio_file)
        midi_file = load_midi(audio_file)
        duration = librosa.core.get_duration(time_series_and_sr[0], time_series_and_sr[1])
        print("audio file:", audio_file)
        print("midi file length:", midi_file.length)
        print("duration:", duration)
        assert abs(int(round(midi_file.length)) - duration) < 0.5, "diff greater than 0.5! omg " \
                                                                   "awfefoje"

def test_if_audio_segments_are_same_shape():
    directory_str_audio = "C:/Users/Lilly/audio_and_midi/audio"
    audio_files_no_duplicates = find_audio_files(directory_str_audio)
    for audio_file in audio_files_no_duplicates:
        time_series_and_sr = load_audio(audio_file)
        sr = time_series_and_sr[1]
        duration_song = librosa.core.get_duration(y=time_series_and_sr[0], sr=sr)
        audio_segment_length = 1
        midi_segment_length = .5
        padding = midi_segment_length / 2
        audio_start_times_missing_first = np.arange(
            padding, (duration_song - midi_segment_length), midi_segment_length)
        # For ex,
        # this will mean that for a song of len 157, the last needed MIDI file would be 156.5-157,
        # so the last audio start time should be 156.25. In this example
        # duration_song-midi_segment_length would be 156.5
        audio_start_times = np.concatenate((np.asarray([0]), audio_start_times_missing_first),
                                           axis=0)
        audio_segment_time_series_first = load_segment_of_audio_and_save(audio_file,
                                                                   audio_start_times[0],
                                                                   audio_segment_length,
                                                                   midi_segment_length,
                                                                   duration_song, sr)
        shape_first_segment = audio_segment_time_series_first.shape
        for start_time in audio_start_times:
            audio_segment_time_series = load_segment_of_audio_and_save(audio_file, start_time,
                                                                       audio_segment_length,
                                                                       midi_segment_length,
                                                                       duration_song, sr)
            assert shape_first_segment == audio_segment_time_series.shape, "segments are not all " \
                                                                           "the same shape!"
        return audio_start_times, audio_segment_length, midi_segment_length

def test_difference_between_audio_first_note_onset_and_midi_first_note_on():
    directory_str_audio = "C:/Users/Lilly/audio_and_midi/audio"
    audio_files_no_duplicates = find_audio_files(directory_str_audio)
    for audio_file in audio_files_no_duplicates:
        time_series_and_sr = load_audio(audio_file)
        first_onset = librosa.onset.onset_detect(time_series_and_sr[0], time_series_and_sr[1],
                                                 units='time')
        midi_file = load_midi(audio_file)
        dumbed_down_midi, ticks_since_start, length_in_secs = create_simplified_midi(midi_file)
        assert abs((first_onset[0] - dumbed_down_midi[0][-1])) < 0.5, \
            "first onset and first dumbed down midi time are significantly different"

def test_if_audio_files_list_has_duplicates():
    directory_str_audio = "C:/Users/Lilly/audio_and_midi/audio"
    audio_files = librosa.util.find_files(directory_str_audio, recurse=False, case_sensitive=True)
    assert len(audio_files) == len(set(audio_files)), "there are duplicates"