import librosa
from mido import MidiFile, Message, MidiTrack
import numpy as np
import ntpath
import time
from collections import defaultdict

#seaborn makes plots prettier
import seaborn
seaborn.set(style='ticks')

#audio playback widget
# from IPython.display import Audio

#TODO: Advice to download with LAME N/A?
def find_audio_files(directory_str_audio):
    audio_files = librosa.util.find_files(directory_str_audio, recurse=False, case_sensitive=True) #recurse False
    # means subfolders are not searched; case_sensitive True ultimately keeps songs from being
    # listed twice
    return audio_files

def load_audio(audio_file):
    floating_point_time_series, sr = librosa.core.load(audio_file, sr=22050*4)
    time_series_and_sr = [floating_point_time_series, sr]

    # with open('filename.pickle', 'wb') as handle:
    #     pickle.dump(time_serieses_and_srs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('filename.pickle', 'rb') as handle:
    #     unserialized_data = pickle.load(handle)

    return time_series_and_sr

#if they're the exact same length and at all out of seguence, nn will never be able to get edges
#doesn't matter bc songs going forward
#good to have room on edges bc frequency analysis for audio is offten weird on edges. bc of
# things like when it sees the start of a wave it doesn't know if it's halfway up the wave or not
#  (this may or may not be a problem)

def timestamps_and_call_load_segment(audio_file, time_series_and_sr):
    sr = time_series_and_sr[1]
    duration_song = librosa.core.get_duration(y=time_series_and_sr[0], sr=sr)
    audio_segment_length = 1
    midi_segment_length = 0.5
    padding = midi_segment_length / 2
    audio_start_times_missing_first = np.arange(
        padding,(duration_song-midi_segment_length), midi_segment_length)
            # For ex,
            # this will mean that for a song of len 157, the last needed MIDI file would be 156.5-157,
            # so the last audio start time should be 156.25. In this example
            # duration_song-midi_segment_length would be 156.5
    audio_start_times = np.concatenate((np.asarray([0]), audio_start_times_missing_first),
                                       axis=0)
    for start_time in audio_start_times:
        audio_segment_time_series = load_segment_of_audio_and_save(audio_file, start_time,
                                                        audio_segment_length,
                                       midi_segment_length, duration_song, sr)

    return audio_start_times, audio_segment_length, midi_segment_length

def silent_audio(sr, padding):
    zeros = int(sr * padding)
    audio = np.zeros((zeros))
    maxv = np.iinfo(np.int16).max #numpy.iinfo(type): Machine limits for integer types.
    silent_audio_time_series = (audio * maxv).astype(np.int16)
    # librosa.output.write_wav("C:/Users/Lilly/audio_and_midi/out_int16.wav", silent_audio_time_series, sr)
    return silent_audio_time_series

def load_segment_of_audio_and_save(audio_file, start, audio_segment_length, midi_segment_length,
                                   duration_song, sr):
    padding = midi_segment_length / 2

    # If you don't want to pad the audio segments, comment out the following code and move the
    # block inside the else condition out of the else condition
    # print("start:", start)
    if start == 0:
        segment_duration = (audio_segment_length - padding)
        audio_segment_time_series_og, sr = librosa.core.load(audio_file, offset=0,
                                                          duration=segment_duration, sr=sr)
        silence = silent_audio(sr, padding)
        audio_segment_time_series = np.concatenate((silence, audio_segment_time_series_og), axis=0)
        # print("audio segment time series:", audio_segment_time_series)
    elif (duration_song - start) < audio_segment_length:
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
    librosa.output.write_wav(filename, audio_segment_time_series, sr)
    # print("end time:", (audio_segment_length + start))
    # print("shape data point:", audio_segment_time_series.shape)
    return audio_segment_time_series

def load_midi(audio_file):
    #Bach is type 1 MIDI, meaning all tracks start at the same time
    midi_str = "C:/Users/Lilly/audio_and_midi/midi/" + ntpath.basename(audio_file)[:-4] + ".mid"
    midi_file = MidiFile(midi_str)
    # print("midi file:", midi_file)
    # last_message = midi_file.tracks[-1][-100:]
    return midi_file

def create_simplified_midi(midi_file):
    simplified_midi = []
    ticks_since_start = 0
    # print("midi file:", midi_file)
    # print("midi file.tracks[-1]:", midi_file.tracks[-1])
    for message in midi_file.tracks[0]:
        if message.type == "set_tempo":
            tempo_in_microsecs_per_beat = message.tempo
    for message in midi_file.tracks[-1]:
        # convert delta time to TICKS since start
        ticks_since_start += message.time
        if message.type == "note_on" or message.type == "note_off":
            simplified_midi.append([message.type, message.note, ticks_since_start])

    # convert ticks since start to absolute seconds
    tempo_in_secs_per_beat = tempo_in_microsecs_per_beat / 1000000
    ticks_per_beat = midi_file.ticks_per_beat
    secs_per_tick = tempo_in_secs_per_beat / ticks_per_beat
    length_in_secs = ticks_since_start * secs_per_tick
    for message in simplified_midi:
        message[-1] = message[-1] / ticks_since_start * length_in_secs
    # For comparison
    midi_length = midi_file.length
    print("midi_file.length:", midi_file.length)
    print("length based on last note off:", length_in_secs)
    return simplified_midi, ticks_since_start, length_in_secs

def chop_simplified_midi(midi_file, midi_segment_length):
    simplified_midi, absolute_ticks_last_note, length_in_secs = create_simplified_midi(midi_file)
    start_timestamps = np.arange(0, length_in_secs, midi_segment_length)
    time_so_far = 0
    midi_segments =[]
    for start_time in start_timestamps:
        midi_segment = []
        for message in simplified_midi:
            end_time = time_so_far + midi_segment_length
            if message[-1] >= time_so_far and message[-1] < end_time:
                midi_segment.append(message)
        midi_segments.append([start_time, midi_segment])
        time_so_far = end_time
    return midi_segments, absolute_ticks_last_note, length_in_secs

def add_note_onsets_to_beginning_when_needed(midi_segments, midi_segment_length):
    #Account for notes which start before a clip and end after a clip. For ex, a note which is
    # from .25 to 1.25. Without this special case accounting, I believe this note would not be
    # present in the simplified MIDI clip
    pitches_to_set_to_on_at_beginning_of_next_segment = []
    for start_time_and_messages in midi_segments:
        start_time = start_time_and_messages[0]
        messages = start_time_and_messages[1]
        for pitch in set(pitches_to_set_to_on_at_beginning_of_next_segment): #TODO CHECK THAT
            # DUPLICATES ARE REMOVED WITH SET
            messages.insert(0, ["note_on", pitch, start_time])

        # Goal is to a dict of dicts like so: {pitch: {num_ons: 4, num_offs: 3}}
        pitch_on_and_off_counts = defaultdict(lambda: defaultdict(int))

        for message in messages:
            pitch = message[1]
            message_type = message[0]
            pitch_on_and_off_counts[pitch][message_type] += 1

        for pitch, on_off_counts_dict in pitch_on_and_off_counts.items():
            count_on = on_off_counts_dict["note_on"]
            count_off = on_off_counts_dict["note_off"]
            # if count_off > count_on:
            #     messages.insert(0, ["note_on", pitch, start_time])
            if count_on > count_off:
                #This is inclusive of the "exclusive ending" (ie This will insert an end time of
                # .5, when technically it should be just under .5)
                pitches_to_set_to_on_at_beginning_of_next_segment.append(pitch)
                end_time = start_time + midi_segment_length
                messages.append(["note_off", pitch, end_time])
    return midi_segments

def reconstruct_midi(midi_file, midi_segments, absolute_ticks_last_note, length_in_secs,
                     midi_segment_length):
    time_so_far = 0
    for midi_segment in midi_segments:
        # time in seconds to absolute ticks
        absolute_ticks_midi_segment = []
        for message in midi_segment[1]:
            absolute_ticks_midi_segment.append([message[0], message[1], message[-1] *
                                                      absolute_ticks_last_note /
                                                      length_in_secs])
        # time in absolute ticks to delta time
        delta_time_midi_segment = []
        for message in absolute_ticks_midi_segment:
            delta_time = int(message[-1] - time_so_far)
            delta_time_midi_segment.append([message[0], message[1], delta_time])
            time_so_far = message[-1]
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        for message in delta_time_midi_segment:
            track.append(Message(message[0], note=message[1], time=message[-1]))
        str_start_time = str(midi_segment[0])
        filename_format = "C:/Users/Lilly/audio_and_midi/segments/midi/{0}_start_time_{1}.mid"
        filename = filename_format.format(midi_file.filename[35:-4], str_start_time)
        mid.save(filename)
    return

def preprocess_audio():
    directory_str_audio = "C:/Users/Lilly/audio_and_midi/audio"
    audio_files = find_audio_files(directory_str_audio)
    for audio_file in audio_files:
        time_series_and_sr = load_audio(audio_file)
        audio_timestamps, audio_segment_length, midi_segment_length = timestamps_and_call_load_segment(audio_file, time_series_and_sr)
        midi_file = load_midi(audio_file)

        #See time differences
        duration = librosa.core.get_duration(time_series_and_sr[0], time_series_and_sr[1])
        midi_len = midi_file.length
        if midi_len != duration:
            print(audio_file, "audio len - midi len:", duration-midi_len)

        midi_segments, absolute_ticks_last_note, length_in_secs = chop_simplified_midi(
            midi_file, midi_segment_length)
        midi_segments_plus_onsets = \
            add_note_onsets_to_beginning_when_needed(midi_segments, midi_segment_length)
        reconstruct_midi(midi_file, midi_segments_plus_onsets, absolute_ticks_last_note,
                         length_in_secs, midi_segment_length)
    print('\007')

def main():
    preprocess_audio()

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

    # "lazy loading" with pickle (lazy in gen. means you don't run it until you need it)
    # TODO: Find song with greatest diff in length and listen to the segments the code creates as is
    # TODO: Check that you're creating the same num of audio and MIDI
    # TODO: CQT audio
    # TODO: one hot encode midi
    # TODO: Define benchmark

    # TODO:
    # Preprocessing audio
    # Audio input encoding to NN (3->1)?
    # Midi encoding to NN (one-hot encode the MIDI output)
    # Make all data the same shape
    # Build NN to learn what midi should be generated from an audio clip
    # Find a way to compare quantitatively the correct midi against the NN generated midi (mean
    # squared, mean absolute (?), r2)

    #TODO: design and record first CNN architecture: I plan to try will be a keras Sequential model. As mentioned above, I will likely place a pooling layer after every one or two convolutional layers. The final layer will match the output dimensions.
    # TODO: compile and train the model.  experiment with multiple numbers of epochs.
    # TODO: test the benchmark and the model with the data reserved for testing and compare the two using the same evaluation metrics.

    # TODO:
    # Parse through note-containing tracks (which are lists of messages)(DONE)
    #



