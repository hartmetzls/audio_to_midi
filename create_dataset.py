import librosa
from mido import MidiFile, Message, MidiTrack
# import librosa.display
import numpy as np
import ntpath
import pandas as pd
import mido
import pickle

#seaborn makes plots prettier
import seaborn
seaborn.set(style='ticks')

#audio playback widget
# from IPython.display import Audio

#TODO: Redo data download with LAME?
def find_audio_files(directory_str_audio):
    audio_files = librosa.util.find_files(directory_str_audio, recurse=False) #recurse False
    # means subfolders are not searched
    ##print (len(audio_files))
    # ##print (audio_files)
    #is ##printing duplicates?f
    audio_files_no_duplicates = []
    for i in range(1,len(audio_files),2):
        audio_files_no_duplicates.append(audio_files[i])
    #print ("num songs:", len(audio_files_no_duplicates))
    return audio_files_no_duplicates

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

def timestamp_array(audio_file, time_series_and_sr):
    duration_song = librosa.core.get_duration(y=time_series_and_sr[0], sr=time_series_and_sr[1])
    audio_segment_length = 1
    midi_segment_length = 0.5
    padding = midi_segment_length / 2
    start = 0
    timestamps = np.arange(0,duration_song+audio_segment_length,padding)
    for timestamp in timestamps:
        load_segment_of_audio_and_save(audio_file, start, audio_segment_length,
                                       midi_segment_length, duration_song)
        start = timestamp - midi_segment_length
    return timestamps, audio_segment_length, midi_segment_length

def silent_audio(sr, midi_segment_length):
    padding = midi_segment_length / 2
    zeros = int(sr * padding)
    audio = np.zeros((zeros)) #TODO: Double check that this var actually specifies length in secs
    maxv = np.iinfo(np.int16).max #numpy.iinfo(type): Machine limits for integer types.
    silent_audio_time_series = (audio * maxv).astype(np.int16)
    # librosa.output.write_wav("C:/Users/Lilly/audio_and_midi/out_int16.wav", silent_audio_time_series, rate)
    return silent_audio_time_series

def load_segment_of_audio_and_save(audio_file, start, audio_segment_length, midi_segment_length,
                                   duration_song):
    padding = midi_segment_length / 2

    # Comment out the following code if you don't want to pad the audio segments
    if start == (-midi_segment_length):
        audio_segment_time_series, sr = librosa.core.load(audio_file, offset=0,
                                                          duration=(audio_segment_length - padding))
        audio_segment_time_series = np.concatenate((silent_audio(sr, midi_segment_length), audio_segment_time_series), axis=0)
    if start == duration_song:
        audio_segment_time_series, sr = librosa.core.load(audio_file, offset=start,
                                                          duration=audio_segment_length - padding)
        audio_segment_time_series = np.concatenate((audio_segment_time_series, silent_audio(sr, midi_segment_length)),axis=0)

    else:
        audio_segment_time_series, sr = librosa.core.load(audio_file, offset=start, duration=audio_segment_length)

    filename = "C:/Users/Lilly/audio_and_midi/segments/audio/" + ntpath.basename(audio_file)[:-4] + str(audio_segment_length + start) + ".wav"
    librosa.output.write_wav(filename, audio_segment_time_series, sr)

def load_midi(audio_file):
    #africa and Bach are both type 1 MIDI, meaning all tracks start at the same time
    midi_str = "C:/Users/Lilly/audio_and_midi/midi/" + ntpath.basename(audio_file)[:-4] + ".mid"
    midi_file = MidiFile(midi_str)
    print("midi file:", midi_file)
    # last_message = midi_file.tracks[-1][-100:]
    return midi_file

def create_dumbed_down_midi(midi_file):
    dumbed_down_midi = []
    ticks_since_start = 0
    print("midi file:", midi_file)
    print("midi file.tracks[-1]:", midi_file.tracks[-1])
    for message in midi_file.tracks[0]:
        if message.type == "set_tempo":
            tempo = message.tempo
    for message in midi_file.tracks[-1]:
        # convert delta time to TICKS since start
        ticks_since_start += message.time
        if message.type == "note_on" or message.type == "note_off":
            dumbed_down_midi.append([message.type, message.note, ticks_since_start])
    length_in_secs = midi_file.length #TODO: Make sure that this is corresponding to a note off and not a meta message?
    print("true length in secs?:", length_in_secs)
    for message in dumbed_down_midi:
        message[-1] = message[-1] / ticks_since_start * length_in_secs
    return dumbed_down_midi, ticks_since_start

def chop_dumbed_down_midi(midi_file, audio_segment_length):
    midi_segment_length = audio_segment_length / 2
    length_in_secs = int(round(midi_file.length)) #if this deosn't work, try length from audio
    stop = length_in_secs+audio_segment_length
    timestamps = np.arange(1,stop,midi_segment_length)
    dumbed_down_midi, absolute_ticks_last_note = create_dumbed_down_midi(midi_file)
    time_so_far = -1
    # TODO: Fix midi clips so that notes hanging over from the previous clip are represented with a note on in their following clip Done, right?
    midi_segments =[]
    for time in timestamps:
        midi_segment = []
        for message in dumbed_down_midi:
            if message[-1] > time_so_far and message[-1] <= time:
                midi_segment.append(message)
        midi_segments.append([time, midi_segment])
        time_so_far = time
    return midi_segments, absolute_ticks_last_note

def add_note_onsets_to_beginning_when_needed(midi_segments):
    for time_and_messages in midi_segments:
        messages = time_and_messages[1]
        #messages is sometimes an empty list
        if len(messages) >= 1:
            cur_time = messages[0][-1]
            notes_to_check = [x[1] for x in messages]
            notes_to_check_no_duplicates = list(dict.fromkeys(notes_to_check))
            for note in notes_to_check_no_duplicates:
                for message in messages:
                    if message[1] == note and message[0] == 'note_off':
                        messages.insert(0,['note_on', note, cur_time])
                        break #TODO: figure out how to do this w/o break
                    if message[1] == note and message[0] == 'note_on':
                        break
    return midi_segments

def reconstruct_midi(midi_file, midi_segments, absolute_ticks_last_note, length_in_secs):
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
        filename = "C:/Users/Lilly/audio_and_midi/segments/midi/" + midi_file.filename[
                                                                      35:-4] + str(
            midi_segment[0]) + ".mid"
        # print("reconstructed mid ticks per beat", mid.ticks_per_beat)
        mid.save(filename)
    return

def preprocess_audio():
    directory_str_audio = "C:/Users/Lilly/audio_and_midi/audio"
    audio_files_no_duplicates = find_audio_files(directory_str_audio)
    for audio_file in audio_files_no_duplicates:
        time_series_and_sr = load_audio(audio_file)
        timestamps, audio_segment_length, midi_segment_length = timestamp_array(audio_file, time_series_and_sr)
        midi_file = load_midi(audio_file)
        midi_segments, absolute_ticks_last_note = chop_dumbed_down_midi(midi_file, audio_segment_length)
        midi_segments_plus_onsets = add_note_onsets_to_beginning_when_needed(midi_segments)
        reconstruct_midi(midi_file, midi_segments_plus_onsets, absolute_ticks_last_note, midi_file.length)


def main():
    # compare_beats("C:/Users/Lilly/audio_and_midi/audio/Bach_BWV871-02_002_20090916-SMD.wav")
    # compare_beats("C:/Users/Lilly/audio_and_midi/audio/Bartok_SZ080-02_002_20110315-SMD.wav")
    preprocess_audio()

if __name__ == '__main__':
    main()

    # dumbed_down_midi = load_midi(audio_files_no_duplicates, fourth_beat_timestamp)

    # fill_in_dumbed_down_midi(dumbed_down_midi)

    # "lazy loading" with pickle (lazy in gen. means you don't run it until you need it)
    # TODO: Define benchmark
    # TODO: Find a way to separate audio/midi files into x num measures or seconds, (segment the audio and MIDI into clips based on seconds (this code is 95% complete))
    # TODO:  pad any short samples (from the end of a track not divisible by n seconds) with silent audio
    # TODO: add silence for the  corresponding MIDI clip .
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



