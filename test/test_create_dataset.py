import pytest
from pytest import approx

from create_dataset import *

# def test_dumbed_down_midi():
#     directory_str_audio = "C:/Users/Lilly/audio_and_midi/audio"
#     audio_files_no_duplicates = find_audio_files(directory_str_audio)
#     for audio_file in audio_files_no_duplicates:
#         # time_series_and_sr = load_audio(audio_file)
#         # every_fourth_timestamp_array = timestamp_array(audio_file, time_series_and_sr)
#         midi_file = load_midi(audio_file)
#         dumbed_down_track, _ = create_dumbed_down_midi(midi_file)
#         previous_message_time = 0
#         for message in dumbed_down_track:
#             cur_message_time = message[2]
#             assert previous_message_time <= cur_message_time, "message out of order"
#             previous_message_time = cur_message_time
#
#         # divide_points = midi_divide_points(midi_file)

# def test_every_fourth_timestamp_array():
#     directory_str_audio = "C:/Users/Lilly/audio_and_midi/audio"
#     audio_files_no_duplicates = find_audio_files(directory_str_audio)
#     for audio_file in audio_files_no_duplicates:
#         time_series_and_sr = load_audio(audio_file)
#         every_fourth_timestamp_array = timestamp_array(audio_file, time_series_and_sr)
#         previous_timestamp = 0
#         for time in every_fourth_timestamp_array:
#             assert previous_timestamp <= time, "time out of order (time out of mind)"
#             previous_timestamp = time

# def test_duration():
#     directory_str_audio = "C:/Users/Lilly/audio_and_midi/audio"
#     audio_files_no_duplicates = find_audio_files(directory_str_audio)
#     for audio_file in audio_files_no_duplicates:
#         time_series_and_sr = load_audio(audio_file)
#         midi_file = load_midi(audio_file)
#         duration = librosa.core.get_duration(time_series_and_sr[0], time_series_and_sr[1])
#         assert int(round(midi_file.length)) == duration, "not equal. ahahfdsakfhdeff!"

