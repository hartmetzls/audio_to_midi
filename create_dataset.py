from mido import Message, MidiTrack, MidiFile
import time
from collections import defaultdict
from cqt import *
from encode_midi_segments import *
from decode_midi import *
from collections import namedtuple
import pickle
import random
random.seed(21)
import ntpath

# data types
AudioDecoded = namedtuple('AudioDecoded', ['time_series', 'sr'])

def find_audio_files(directory_str):
    directory_str_audio = directory_str + "audio"
    audio_files = librosa.util.find_files(directory_str_audio, recurse=False, case_sensitive=True) #recurse False
    # means subfolders are not searched; case_sensitive True ultimately keeps songs from being
    # listed twice
    return audio_files

def load_audio(audio_file):
    time_series, sr = librosa.core.load(audio_file, sr=22050*4)
    time_series_and_sr = AudioDecoded(time_series, sr)
    return time_series_and_sr

def audio_timestamps(audio_file, time_series_and_sr):
    sr = time_series_and_sr[1]
    duration_song = librosa.core.get_duration(y=time_series_and_sr[0], sr=sr)
    audio_segment_length = 1
    midi_segment_length = 0.5
    padding = midi_segment_length / 2
    audio_start_times_missing_first = np.arange(
        padding,(duration_song-midi_segment_length), midi_segment_length)
            # For ex,
            # this will mean that for a song of len 157 seconds, the last needed MIDI file would be 156.5-157,
            # so the last audio start time should be 156.25. In this example
            # duration_song minus midi_segment_length is 156.5
    audio_start_times = np.concatenate((np.asarray([0]), audio_start_times_missing_first),
                                       axis=0)
    return audio_start_times, audio_segment_length, midi_segment_length

def silent_audio(sr, padding):
    zeros = int(sr * padding)
    audio = np.zeros((zeros))
    maxv = np.iinfo(np.int16).max #numpy.iinfo(type): Machine limits for integer types.
    silent_audio_time_series = (audio * maxv).astype(np.int16)
    return silent_audio_time_series

# padding_portion provides the option to set padding to zero, so that if a better temporally aligned dataset is used
# in the future, the code can segmented without padding
def load_segment_of_audio_and_save(audio_file, start, audio_segment_length, midi_segment_length, duration_song, sr,
                                   padding_portion=.5):
    padding = midi_segment_length * padding_portion
    # If you don't want to pad the audio segments, comment out the following code and move the
    # line inside the else condition out of the else condition
    if start == 0:
        segment_duration = (audio_segment_length - padding)
        audio_segment_time_series_og, sr = librosa.core.load(audio_file, offset=0,
                                                          duration=segment_duration, sr=sr)
        silence = silent_audio(sr, padding)
        audio_segment_time_series = np.concatenate((silence, audio_segment_time_series_og), axis=0)
    elif (duration_song - start) < audio_segment_length: #if we're at the end of a song and need
        # padding
        segment_duration = (duration_song - start)
        audio_segment_time_series_og, sr = librosa.core.load(audio_file, offset=start,
                                                          duration=segment_duration, sr=sr)
        audio_segment_time_series = np.concatenate(
            (audio_segment_time_series_og, silent_audio(sr, (audio_segment_length-segment_duration))),axis=0)
    else:
        audio_segment_time_series, sr = librosa.core.load(audio_file, offset=start,
                                                          duration=audio_segment_length, sr=sr)
    filename_format = "C:/Users/Lilly/audio_and_midi/segments/audio/{0}_start_time_{1}.wav"
    filename = filename_format.format(ntpath.basename(audio_file)[:-4], str(start))

    # for testing by listening to audio (currently written for windows)
    # librosa.output.write_wav(filename, audio_segment_time_series, sr)

    return audio_segment_time_series

def load_midi(directory_str, audio_file):
    midi_base_str = directory_str + "midi/"
    midi_str = midi_base_str + ntpath.basename(audio_file)[:-4] + ".mid"
    midi_file = MidiFile(midi_str)
    return midi_file

def create_simplified_midi(midi_file):
    simplified_midi = []
    ticks_since_start = 0

    # debugging/for future use with a dataset other than Saarland Music Dataset:
    # check for note offs without note ons and check control changes in song
    print("midi file:", midi_file)
    count_ons = 0
    count_offs = 0
    control_changes = []

    for message in midi_file.tracks[0]:
        if message.type == "set_tempo":
            tempo_in_microsecs_per_beat = message.tempo
    for message in midi_file.tracks[-1]:
        # convert delta time (delta ticks) to TICKS since start
        ticks_since_start += message.time
        if message.type == "note_on" or message.type == "note_off":
            simplified_midi.append([message.type, message.note, ticks_since_start])

            # debugging/for future use with a dataset other than SMD
            if message.type == "note_on":
                count_ons += 1
            else:
                count_offs += 1

        # debugging/for future use with a dataset other than SMD: check control changes in song; check for 123
        if message.type == 'control_change':
            control_changes.append(message.control)
            if message.control == 123:
                print("CONTROL ALL NOTES OFF MESSAGE!")

    # debugging/for future use with a dataset other than SMD
    if count_ons != count_offs:
        print("INEQUAL NUM ONS AND OFFS (prior to simplified midi)")
        print("count ons:", count_ons)
        print("offs:", count_offs)

    # convert ticks since start to absolute seconds
    tempo_in_secs_per_beat = tempo_in_microsecs_per_beat / 1000000
    ticks_per_beat = midi_file.ticks_per_beat
    secs_per_tick = tempo_in_secs_per_beat / ticks_per_beat
    length_in_secs_full_song = ticks_since_start * secs_per_tick
    for message in simplified_midi:
        message[-1] = message[-1] / ticks_since_start * length_in_secs_full_song

    # For comparison
    midi_length = midi_file.length
    # print("midi_file.length:", midi_file.length)
    # print("length based on last note off:", length_in_secs_full_song)

    # debugging/for future use with a dataset other than SMD
    print(set(control_changes))

    return simplified_midi, ticks_since_start, length_in_secs_full_song

def chop_simplified_midi(midi_file, midi_segment_length, simplified_midi, absolute_ticks_last_note, midi_start_times):

    # debugging/for future use with a dataset other than SMD: check for note ons and offs equal
    count_note_on = 0
    count_note_off = 0
    for message in simplified_midi:
        message_type = message[0]
        if message_type == 'note_on':
            count_note_on += 1
        if message_type == 'note_off':
            count_note_off += 1
    if count_note_on != count_note_off:
        print("midi file:", midi_file)
        print("inequal num ons and offs")

    # erase a redundant note_on or a redundant note off of the same pitch
    pitches_on = []
    messages_to_erase = []
    for message in simplified_midi:
        message_type = message[0]
        pitch = message[1]
        if message_type == 'note_on':
            if pitch in pitches_on:
                messages_to_erase.append(message)
            else:
                pitches_on.append(pitch)
        elif message_type == 'note_off':
            if pitch in pitches_on:
                pitches_on.remove(pitch)
            else:
                print("redundant off message found")
                messages_to_erase.append(message)
    for message in messages_to_erase:
        simplified_midi.remove(message)

    time_so_far = 0
    midi_segments =[]
    for midi_start_time in midi_start_times:
        midi_segment = []
        for message in simplified_midi:
            end_time = time_so_far + midi_segment_length
            if message[-1] >= time_so_far and message[-1] < end_time:
                midi_segment.append(message)
        midi_segments.append([midi_start_time, midi_segment])
        time_so_far = end_time
    return midi_segments, absolute_ticks_last_note

def add_note_onsets_to_beginning_when_needed(midi_segments, midi_segment_length):
    # account for notes which start before a segment and end after a segment. For ex, a note which is
    # from .25 to 1.25. Without this special case accounting, this note would not be
    # present in the simplified MIDI clip
    pitches_to_set_to_on_at_beginning_of_segment = []
    for start_time_and_messages in midi_segments:
        start_time = start_time_and_messages[0]
        messages = start_time_and_messages[1]
        for pitch in pitches_to_set_to_on_at_beginning_of_segment:
            messages.insert(0, ["note_on", pitch, start_time])
        pitches_to_set_to_on_at_beginning_of_segment = []
        # goal is to build a dict of dicts like so: {pitch: {num_ons: 4, num_offs: 3}}
        pitch_on_and_off_counts = defaultdict(lambda: defaultdict(int))

        for message in messages:
            pitch = message[1]
            message_type = message[0]
            pitch_on_and_off_counts[pitch][message_type] += 1

        for pitch, on_off_counts_dict in pitch_on_and_off_counts.items():
            count_on = on_off_counts_dict["note_on"]
            count_off = on_off_counts_dict["note_off"]
            if count_on > count_off:
                # this includes the "exclusive ending" (ie This will insert an end time of
                # .5, when technically it should be just under .5)
                pitches_to_set_to_on_at_beginning_of_segment.append(pitch)
                end_time = start_time + midi_segment_length
                messages.append(["note_off", pitch, end_time])
    return midi_segments

def find_lowest_and_highest_midi_note_numbers(simplified_midi):
    lowest_note_number_so_far = 127
    highest_note_number_so_far = 0
    for message in simplified_midi:
        note_number = message[1]
        if note_number < lowest_note_number_so_far:
            lowest_note_number_so_far = note_number
        if note_number > highest_note_number_so_far:
            highest_note_number_so_far = note_number
    return lowest_note_number_so_far, highest_note_number_so_far

def reconstruct_midi(midi_filename, midi_segments, absolute_ticks_last_note, length_in_secs_full_song):
    time_so_far = 0
    for midi_segment in midi_segments:

        # time in seconds to absolute ticks
        absolute_ticks_midi_segment = []
        start_time = midi_segment[0]
        messages = midi_segment[1]
        for message in messages:
            note_on_or_off = message[0]
            pitch = message[1]
            scaled_time = message[-1]
            time = scaled_time + start_time
            absolute_ticks = time * absolute_ticks_last_note / length_in_secs_full_song
            absolute_ticks_midi_segment.append([note_on_or_off, pitch, absolute_ticks])
        # time in absolute ticks to delta time
        delta_time_midi_segment = []
        for message in absolute_ticks_midi_segment:
            note_on_or_off = message[0]
            pitch = message[1]
            time = message[-1]
            delta_time = int(time - time_so_far)
            delta_time_midi_segment.append([note_on_or_off, pitch, delta_time])
            time_so_far = time

        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        for message in delta_time_midi_segment:
            note_on_or_off = message[0]
            pitch = int(message[1])
            delta_ticks = message[-1]

            # debugging/for future use with a dataset other than SMD
            if type(delta_ticks) != int or delta_ticks < 0:
                print("time issue")

            track.append(Message(note_on_or_off, note=pitch, time=delta_ticks))

        # for testing by listening to midi (currently written for windows)
        # str_start_time = str(midi_segment[0])
        # filename_format = "C:/Users/Lilly/audio_and_midi/segments/midi/{0}_start_time_{1}.mid"
        # filename = filename_format.format(midi_filename, str_start_time)
        # mid.save(filename)

    return

def done_beep():
    import os
    # Windows
    if os.name == 'nt':
        import winsound
        duration = 1500  # millisecond
        freq = 392  # Hz
        winsound.Beep(freq, duration)
    # Linux
    if os.name == 'posix':
        duration = 1  # second
        freq = 392 # Hz
        os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))

def preprocess_audio_and_midi(directory_str):
    audio_files = find_audio_files(directory_str)
    cqt_segments = []
    all_songs_encoded_midi_segments = []
    midi_segments_count = 0
    for audio_file in audio_files:
        audio_decoded = load_audio(audio_file)
        time_series, sr = audio_decoded
        audio_start_times, audio_segment_length, midi_segment_length = audio_timestamps(
            audio_file, audio_decoded)
        midi_file = load_midi(directory_str, audio_file)
        duration_song = librosa.core.get_duration(time_series, sr)

        # print time differences
        midi_len = midi_file.length
        if midi_len != duration_song:
            print(audio_file, "audio len - midi len:", duration_song-midi_len)

        simplified_midi, absolute_ticks_last_note, length_in_secs_full_song = create_simplified_midi(
            midi_file)
        lowest_midi_note, highest_midi_note = find_lowest_and_highest_midi_note_numbers(simplified_midi)
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
            padding_portion = .5  # padding on each side of audio is midi_segment_length * padding portion
            audio_segment_time_series = load_segment_of_audio_and_save(audio_file, start_time,
                                                                       audio_segment_length,
                                                                       midi_segment_length,
                                                                       duration_song, sr,
                                                                       padding_portion)
            cqt_of_segment = audio_segments_cqt(audio_segment_time_series, sr)
            cqt_segments.append(cqt_of_segment)

        midi_segments, absolute_ticks_last_note = chop_simplified_midi(midi_file, midi_segment_length, simplified_midi, absolute_ticks_last_note, midi_start_times)
        midi_start_times_and_segments_incl_onsets = \
            add_note_onsets_to_beginning_when_needed(midi_segments, midi_segment_length)

        for midi_segment in midi_start_times_and_segments_incl_onsets:
            midi_start_time = midi_segment[0]
            messages = midi_segment[1]
            encoded_segment, num_discrete_time_values, num_notes = encode_midi_segment(midi_start_time, messages, midi_segment_length, lowest_midi_note, highest_midi_note)
            all_songs_encoded_midi_segments.append(encoded_segment)

            # debugging equal num cqt and midi segment
            midi_segments_count += 1

        # # testing
        # # encoded_segment = all_songs_encoded_midi_segments[1]
        # decoded_segments = []
        # for encoded_segment in all_songs_encoded_midi_segments:
        #     decoded_midi = decode_midi_segment(encoded_segment, midi_segment_length, num_discrete_time_values, lowest_midi_note)
        #     decoded_segments.append(decoded_midi)
        # decoded_midi_start_times_and_segments = []
        # for i in range(len(midi_start_times)):
        #     decoded_midi_start_times_and_segments.append([midi_start_times[i], decoded_segments[i]])

        # code for midi reconstruction (for listening purposes)
        # midi_filename = midi_file.filename[35:-4]
        # reconstruct_midi(midi_filename, decoded_midi_start_times_and_segments, absolute_ticks_last_note, length_in_secs_full_song)

    # pickle time
    with open('cqt_segments_midi_segments.pkl', 'wb') as handle:
        pickle.dump(cqt_segments, handle)
        pickle.dump(all_songs_encoded_midi_segments, handle)
    return

def main():
    # You must set directory str to be the filepath to your directory containing 2 directories: one named "audio"
    # (containing the
    # audio files) and one named "midi" (containing the MIDI files). For example, "C:/Users/Lilly/audio_and_midi/" or
    # "/home/lilly/Downloads/audio_midi/"
    directory_str = "/home/lilly/Downloads/audio_midi/"
    preprocess_audio_and_midi(directory_str)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))






