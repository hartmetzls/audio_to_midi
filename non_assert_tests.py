from create_dataset import *
import os
from os.path import exists
import shutil
from models import pickle_if_not_pickled

#NON-ASSERT TESTS
def difference_between_audio_first_note_onset_and_midi_first_note_on():
    directory_str_audio = "C:/Users/Lilly/audio_and_midi/audio"
    audio_files_no_duplicates = find_audio_files(directory_str_audio)
    for audio_file in audio_files_no_duplicates:
        time_series_and_sr = load_audio(audio_file)
        first_onset = librosa.onset.onset_detect(time_series_and_sr[0], time_series_and_sr[1],
                                                 units='time')
        midi_file = load_midi(audio_file)
        simplified_midi, ticks_since_start, length_in_secs = create_simplified_midi(midi_file)
        # assert (first_onset[0] - simplified_midi[0][-1]) == 0, "first onset and first dumbed " \
        #                                                         "down " \
        #                                                    "midi time are not the same"
        print("midi file:", midi_file)
        print("est'd first onset in audio - first onset in MIDI:",(first_onset[0] -
                                                                simplified_midi[0][-1]))

def find_song_with_greatest_diff_in_length():
    directory_str_audio = "C:/Users/Lilly/audio_and_midi/audio"
    audio_files_no_duplicates = find_audio_files(directory_str_audio)
    greatest_diff_yet = [None, 0]
    for audio_file in audio_files_no_duplicates:
        time_series_and_sr = load_audio(audio_file)
        midi_file = load_midi(audio_file)
        duration = librosa.core.get_duration(time_series_and_sr[0], time_series_and_sr[1])
        print("audio file:", audio_file)
        print("midi file length:", midi_file.length)
        print("duration:", duration)
        diff = abs(int(round(midi_file.length)) - duration)
        if diff > greatest_diff_yet[1]:
            greatest_diff_yet = [audio_file, diff]
    print("song with biggest diff:", greatest_diff_yet)

def make_test_midi():
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    #Test note off without on
    track.append(Message("note_off", note=64, time=640))
    #Add control note
    track.append(Message("note_on", note=69, time=1000))
    track.append(Message("note_off", note=69, time=1100))
    #Test note on without off
    track.append(Message("note_on", note=73, time=1280))
    filename_format = "C:/Users/Lilly/audio_and_midi/segments/midi/{0}_start_time_{1}.mid"
    filename = filename_format.format("testing_note_on_off", "test_1")
    mid.save(filename)
    return

def check_if_there_is_a_midi_file_for_every_audio_file():
    directory_str_audio = "C:/Users/Lilly/audio_and_midi/segments/audio"
    audio_files = find_audio_files(directory_str_audio)
    midi_file_folder = "C:/Users/Lilly/audio_and_midi/segments/midi/"
    count = 0
    to_listen_to = []
    for audio_file in audio_files:
        basename = ntpath.basename(audio_file)[:-4]
        index_last_underscore = basename.rfind("_")
        audio_time = basename[index_last_underscore+1:]
        song_title = basename[:index_last_underscore]
        midi_time = float(audio_time) + 0.25
        midi_file_name = song_title + "_" + str(midi_time) + ".mid"
        if audio_time != "0.0":
            if not exists(midi_file_folder + midi_file_name):
                to_listen_to.append(audio_file)
                count += 1
    print(count)
    new_dir = "C:/Users/Lilly/audio_and_midi/segments/no_corresponding_midi"
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    for file in to_listen_to:
        shutil.copy(file, new_dir)

    done_beep()

def check_if_there_is_an_audio_file_for_every_midi_file():
    directory_str_midi = "C:/Users/Lilly/audio_and_midi/segments/midi"
    midi_files = os.listdir(directory_str_midi)
    audio_file_folder = "C:/Users/Lilly/audio_and_midi/segments/audio/"
    count = 0
    to_listen_to = []
    for midi_file in midi_files:
        basename = ntpath.basename(midi_file)[:-4]
        index_last_underscore = basename.rfind("_")
        midi_time = basename[index_last_underscore+1:]
        song_title = basename[:index_last_underscore]
        audio_time = float(midi_time) - 0.25
        audio_file_name = song_title + "_" + str(audio_time) + ".wav"
        if midi_time != "0.0":
            if not exists(audio_file_folder + audio_file_name):
                to_listen_to.append(midi_file)
                count += 1
    print(count)
    new_dir = "C:/Users/Lilly/audio_and_midi/segments/no_corresponding_audio"
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    for file in to_listen_to:
        complete_file = directory_str_midi + "/" + file
        shutil.copy(complete_file, new_dir)

    done_beep()

def check_reconstruct_midi_empty_begin():
    filename = "test_case_empty_begin"
    midi_segments = [[0.0, []]]
    #[[0.0, []],[0.5, ['note_on', 64, 0.75], ['note_off', 64, 0.99]]]
    absolute_ticks_last_note = 3000
    length_in_secs_full_song = 1
    reconstruct_midi(filename, midi_segments, absolute_ticks_last_note, length_in_secs_full_song)

def check_reconstruct_midi_start_after_song_end():
    filename = "test_case_start_after_song_end"
    midi_segments = [[100.0, []]]
    #[[0.0, []],[0.5, ['note_on', 64, 0.75], ['note_off', 64, 0.99]]]
    absolute_ticks_last_note = 3000
    length_in_secs_full_song = 1.5
    reconstruct_midi(filename, midi_segments, absolute_ticks_last_note, length_in_secs_full_song)

def check_reconstruct_midi_start_after_song_end_non_empty():
    filename = "test_case_start_after_song_end_non_empty"
    # midi_segments = [[0.0, []],[0.5, ['note_on', 64, 0.75], ['note_off', 64, 0.99]]]
    midi_segments = [[100.0, [['note_on', 64, 101.75]]]]
    #[[0.0, []],[0.5, ['note_on', 64, 0.75], ['note_off', 64, 0.99]]]
    absolute_ticks_last_note = 3000
    length_in_secs_full_song = 1.5
    reconstruct_midi(filename, midi_segments, absolute_ticks_last_note, length_in_secs_full_song)

def check_reconstruct_midi_start_after_song_end_non_empty2():
    filename = "test_case_start_after_song_end_non_empty2"
    # midi_segments = [[0.0, []],[0.5, ['note_on', 64, 0.75], ['note_off', 64, 0.99]]]
    midi_segments = [[100.0, [['note_on', 64, 1.75]]]]
    #[[0.0, []],[0.5, ['note_on', 64, 0.75], ['note_off', 64, 0.99]]]
    absolute_ticks_last_note = 3000
    length_in_secs_full_song = 105.5
    reconstruct_midi(filename, midi_segments, absolute_ticks_last_note, length_in_secs_full_song)

def catch_audio_no_midi_or_vice_versa():
    directory_str_audio = "C:/Users/Lilly/audio_and_midi/audio"
    audio_files = find_audio_files(directory_str_audio)
    biggest_diff_in_num_start_times = 0
    num_songs_with_start_time_diff = 0
    for audio_file in audio_files:
        time_series_and_sr = load_audio(audio_file)
        audio_start_times, audio_segment_length, midi_segment_length = audio_timestamps(
            audio_file, time_series_and_sr)
        midi_file = load_midi(audio_file)

        # See time differences
        # duration = librosa.core.get_duration(time_series_and_sr[0], time_series_and_sr[1])
        # midi_len = midi_file.length
        # if midi_len != duration:
        #     print(audio_file, "audio len - midi len:", duration - midi_len)

        midi_segments, absolute_ticks_last_note, length_in_secs_full_song = chop_simplified_midi(
            midi_file, midi_segment_length)
        midi_start_timestamps = np.arange(0, length_in_secs_full_song, midi_segment_length)

        audio_starts_minus_midi_starts = len(audio_start_times) - len(midi_start_timestamps)
        if abs(audio_starts_minus_midi_starts) > 0:
            num_songs_with_start_time_diff += 1
            if abs(audio_starts_minus_midi_starts) > abs(biggest_diff_in_num_start_times):
                biggest_diff_in_num_start_times = audio_starts_minus_midi_starts
                print("last_biggest_diff_in_num_start_times:", audio_file)
                print("len audio timestamps:", len(audio_start_times))
                print("len midi timestamps:", len(midi_start_timestamps))
        print("num songs with diff num start times:", num_songs_with_start_time_diff)
        midi_segments_plus_onsets = \
            add_note_onsets_to_beginning_when_needed(midi_segments, midi_segment_length)
        midi_filename = midi_file.filename[35:-4]
        reconstruct_midi(midi_filename, midi_segments_plus_onsets, absolute_ticks_last_note,
                         length_in_secs_full_song)

def run_without_errors():
    directory_str_audio = "C:/Users/Lilly/audio_and_midi/audio"
    audio_files = find_audio_files(directory_str_audio)
    cqt_segments = []
    all_songs_encoded_midi_segments = []
    midi_segments_count = 0
    for audio_file in audio_files:
        audio_decoded = load_audio(audio_file)
        time_series, sr = audio_decoded
        audio_start_times, audio_segment_length, midi_segment_length = audio_timestamps(
            audio_file, audio_decoded)
        midi_file = load_midi(audio_file)

        duration_song = librosa.core.get_duration(time_series, sr)

        # See time differences
        midi_len = midi_file.length
        if midi_len != duration_song:
            print(audio_file, "audio len - midi len:", duration_song - midi_len)

        simplified_midi, absolute_ticks_last_note, length_in_secs_full_song = create_simplified_midi(
            midi_file)

        lowest_midi_note, highest_midi_note = find_lowest_and_highest_midi_note_numbers(
            simplified_midi)

        midi_start_times = np.arange(0, length_in_secs_full_song, midi_segment_length)

        if len(audio_start_times) != len(midi_start_times):
            if len(audio_start_times) > len(midi_start_times):
                num_start_times = len(midi_start_times)
                audio_start_times_shortened = audio_start_times[:num_start_times]
                audio_start_times = audio_start_times_shortened
            else:
                num_start_times = len(audio_start_times)
                midi_start_times_shortened = midi_start_times[:num_start_times]
                midi_start_times = midi_start_times_shortened

        for start_time in audio_start_times:
            padding_portion = .5  # padding on each side of audio is midi_segment_length *
            # padding portion
            audio_segment_time_series = load_segment_of_audio_and_save(audio_file, start_time,
                                                                       audio_segment_length,
                                                                       midi_segment_length,
                                                                       duration_song, sr,
                                                                       padding_portion)
            cqt_of_segment = audio_segments_cqt(audio_segment_time_series, sr)
            cqt_segments.append(cqt_of_segment)

        midi_segments, absolute_ticks_last_note = chop_simplified_midi(midi_file,
                                                                       midi_segment_length,
                                                                       simplified_midi,
                                                                       absolute_ticks_last_note,
                                                                       midi_start_times)
        midi_start_times_and_segments_incl_onsets = \
            add_note_onsets_to_beginning_when_needed(midi_segments, midi_segment_length)

        for midi_segment in midi_start_times_and_segments_incl_onsets:
            midi_start_time = midi_segment[0]
            messages = midi_segment[1]
            encoded_segment, num_discrete_time_values, num_notes = encode_midi_segment(
                midi_start_time, messages,
                midi_segment_length, lowest_midi_note, highest_midi_note)
            all_songs_encoded_midi_segments.append(encoded_segment)  # TODO: testcorrect num

            # debugging equal num cqt and midi segment
            midi_segments_count += 1

        # here for testing
        for midi in all_songs_encoded_midi_segments:
            decoded_midi = decode_midi_segment(midi, midi_segment_length,
                                           num_discrete_time_values, lowest_midi_note)

def min_max_median_mean():
    cqt_segments, midi_segments = pickle_if_not_pickled()
    cqt_segments_array = np.array(cqt_segments)
    cqt_min = np.min(cqt_segments_array)
    cqt_max = np.max(cqt_segments_array)
    cqt_median = np.median(cqt_segments_array)
    cqt_mean = np.mean(cqt_segments_array)
    midi_segments_array = np.array(midi_segments)
    midi_median = np.median(midi_segments_array)
    midi_mean = np.mean(midi_segments_array)
    print(cqt_mean, midi_mean)





def main():
    # difference_between_audio_first_note_onset_and_midi_first_note_on()
    # find_song_with_greatest_diff_in_length()
    # make_test_midi()
    # check_if_there_is_a_midi_file_for_every_audio_file()
    # check_if_there_is_an_audio_file_for_every_midi_file()
    # check_reconstruct_midi_empty_begin()
    # check_reconstruct_midi_start_after_song_end()
    # check_reconstruct_midi_start_after_song_end_non_empty()
    # check_reconstruct_midi_start_after_song_end_non_empty2()
    # catch_audio_no_midi_or_vice_versa()
    # find_lowest_and_highest_midi_note_numbers()
    # run_without_errors()
    min_max_median_mean()

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))