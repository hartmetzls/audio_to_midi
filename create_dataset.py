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

def count_beats_audio(time_series_and_sr):
    # ##print(time_series_and_sr.shape) #I believe all songs are loaded as mono by default

    # estimated beat event locations in the specified units (default is frame indices)
    # TODO: CAN'T USE BEAT DETECTOR' -> TIME. pad with silent audio and silent midi when needed ~
    #  .5 secs
    tempo, estimated_beat_event_locations = librosa.beat.beat_track(y=time_series_and_sr[0], sr=time_series_and_sr[1])
    return len(estimated_beat_event_locations)

def find_fourth_beats(audio_file, time_series_and_sr):
    # ##print(time_series_and_sr.shape) #I believe all songs are loaded as mono by default

    # estimated beat event locations in the specified units (default is frame indices)
    tempo, estimated_beat_event_locations = librosa.beat.beat_track(y=time_series_and_sr[0], sr=time_series_and_sr[1])

    # #Or ##print estimated beat event locations as timestamps
    beat_timestamps = librosa.frames_to_time(estimated_beat_event_locations, sr=time_series_and_sr[1])
    every_fourth_timestamp_array = beat_timestamps[3::4]
    every_fourth_timestamp_array = np.append(every_fourth_timestamp_array,
                                             (librosa.core.get_duration(y=time_series_and_sr[0],
                                                                     sr=time_series_and_sr[1])))
    #print("every fourth timestamp:")
    #print(every_fourth_timestamp_array)
    start = 0
    for timestamp in every_fourth_timestamp_array:
        load_four_beats_of_audio_and_save(audio_file, start, (timestamp - start))
        start = timestamp
    return every_fourth_timestamp_array

def load_four_beats_of_audio_and_save(audio_file, start, duration):
    four_beats_audio_time_series, sr = librosa.core.load(audio_file, offset=start, duration=duration)
    filename = "C:/Users/Lilly/audio_and_midi/four_beats/audio/" + ntpath.basename(audio_file)[:-4] + str(duration + start) + ".wav"
    librosa.output.write_wav(filename, four_beats_audio_time_series, sr)

def load_midi(audio_file):
    #africa and Bach are both type 1 MIDI, meaning all tracks start at the same time
    midi_str = "C:/Users/Lilly/audio_and_midi/midi/" + ntpath.basename(audio_file)[:-4] + ".mid"
    midi_file = MidiFile(midi_str)
    print("midi file:", midi_file)
    # last_message = midi_file.tracks[-1][-100:]
    return midi_file

def count_beats_midi(midi_file):
    dumbed_down_midi = []
    ticks_since_start = 0
    print("midi file:", midi_file)
    print("midi file.tracks[-1]:", midi_file.tracks[-1])
    for message in midi_file.tracks[-1]:
        #convert delta time to TICKS since start
        ticks_since_start += message.time
        if message.type == "note_on" or message.type == "note_off":
            dumbed_down_midi.append([message.type, message.note, ticks_since_start])
    end_ticks_since_start = dumbed_down_midi[-1][2]
    num_beats = end_ticks_since_start / midi_file.ticks_per_beat
    return num_beats

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
    # TODO: Ask Lou: because ticks_since_start includes the message.time of ALL messages, isn't using midi_file.length okay?
    for message in dumbed_down_midi:
        message[-1] = message[-1] / ticks_since_start * length_in_secs
    return dumbed_down_midi, ticks_since_start

def chop_dumbed_down_midi(midi_file, every_fourth_timestamp_array):
    dumbed_down_midi, absolute_ticks_last_note = create_dumbed_down_midi(midi_file)
    length_in_secs = midi_file.length
    time_so_far = -1
    # TODO: Fix midi clips so that notes hanging over from the previous clip are represented with a note on in their following clip
    four_beat_chunks =[]
    for time in every_fourth_timestamp_array:
        four_beats_of_midi = []
        for message in dumbed_down_midi:
            if message[-1] > time_so_far and message[-1] <= time:
                four_beats_of_midi.append(message)
        four_beat_chunks.append([time, four_beats_of_midi])
        time_so_far = time
    return four_beat_chunks, absolute_ticks_last_note

def add_note_onsets_to_beginning_when_needed(four_beat_chunks):
    for time_and_messages in four_beat_chunks:
        messages = time_and_messages[1]
        #messages is sometimes an empty list #TODO: Does this effect the four beat chunks outcome?
        if len(messages) >= 1:
            cur_time = messages[0][-1]
            notes_to_check = [x[1] for x in messages]
            notes_to_check_no_duplicates = list(dict.fromkeys(notes_to_check))
            for note in notes_to_check_no_duplicates:
                for message in messages:
                    if message[1] == note and message[0] == 'note_off':
                        messages.insert(0,['note_on', note, cur_time])
                        break
                    if message[1] == note and message[0] == 'note_on':
                        break
    return four_beat_chunks

def reconstruct_midi(midi_file, four_beat_chunks, absolute_ticks_last_note, length_in_secs):
        # #Let's reconstruct some shit
        #print("IN RECONSTRUCTION")
    time_so_far = 0
    for four_beats_of_midi in four_beat_chunks:
        if len(four_beats_of_midi[1]) >= 1:
            # time in seconds to absolute ticks
            absolute_ticks_four_beats_of_midi = []
            for message in four_beats_of_midi[1]:
                absolute_ticks_four_beats_of_midi.append([message[0], message[1], message[-1] *
                                                          absolute_ticks_last_note /
                                                          length_in_secs])
            absolute_ticks_so_far = absolute_ticks_four_beats_of_midi[-1][-1]
            # time in absolute ticks to delta time
            delta_time_four_beats_of_midi = []
            for message in absolute_ticks_four_beats_of_midi:
                delta_time = int(message[-1] - time_so_far)
                delta_time_four_beats_of_midi.append([message[0], message[1], delta_time])
                time_so_far = message[-1]
            mid = MidiFile()
            track = MidiTrack()
            mid.tracks.append(track)
            for message in delta_time_four_beats_of_midi:
                track.append(Message(message[0], note=message[1], time=message[-1]))
            filename = "C:/Users/Lilly/audio_and_midi/four_beats/midi/" + midi_file.filename[
                                                                          35:-4] + str(
                four_beats_of_midi[0]) + ".mid"
            # print("reconstructed mid ticks per beat", mid.ticks_per_beat)
            mid.save(filename)
    return

def midi_divide_points(midi_file):
    last_tick = 0
    for message in midi_file.tracks[-1]:
        if message[-1] > last_tick:
            last_tick = message[-1]
    print("last tick:", last_tick)
    divide_points = []
    for i in range(0,last_tick,1920):
        divide_points.append(i)
    return divide_points

def preprocess_audio():
    directory_str_audio = "C:/Users/Lilly/audio_and_midi/audio"
    audio_files_no_duplicates = find_audio_files(directory_str_audio)
    for audio_file in audio_files_no_duplicates:
        time_series_and_sr = load_audio(audio_file)
        every_fourth_timestamp_array = find_fourth_beats(audio_file, time_series_and_sr)
        midi_file = load_midi(audio_file)
        four_beat_chunks, absolute_ticks_last_note = chop_dumbed_down_midi(midi_file, every_fourth_timestamp_array)
        four_beat_chunks_plus_onsets = add_note_onsets_to_beginning_when_needed(four_beat_chunks)
        reconstruct_midi(midi_file, four_beat_chunks_plus_onsets, absolute_ticks_last_note, midi_file.length)
        # divide_points = midi_divide_points(midi_file)

def compare_beats(audio_file):
    time_series_and_sr = load_audio(audio_file)
    beats_audio = count_beats_audio(time_series_and_sr)
    print("beats audio:", beats_audio)
    midi_file = load_midi(audio_file)
    beats_midi = count_beats_midi(midi_file)
    print("beats midi:{0}, then messages: {1}".format(beats_midi, "go away"))

# def midi_length_in_secs(midi_file):

def main():
    # compare_beats("C:/Users/Lilly/audio_and_midi/audio/Bach_BWV871-02_002_20090916-SMD.wav")
    # compare_beats("C:/Users/Lilly/audio_and_midi/audio/Bartok_SZ080-02_002_20110315-SMD.wav")
    preprocess_audio()

if __name__ == '__main__':
    main()

    # dumbed_down_midi = load_midi(audio_files_no_duplicates, fourth_beat_timestamp)
    # four_beats_of_audio = load_four_beats_of_audio_and_save(audio_files_no_duplicates, fourth_beat_timestamp)

    # fill_in_dumbed_down_midi(dumbed_down_midi)

    # TODO: submit proposal for audio to midi
    # TODO:

    # change to one beat samples instead of four
    # "lazy loading" with pickle (lazy in gen. means you don't run it until you need it)
    # Define benchmark, Write proposal
    # Find dataset of audio w/ corresponding midi file
    # Find a way to separate audio/midi files into x num measures or seconds,
    # Preprocessing audio
    # Audio input encoding to NN
    # Midi encoding to NN
    # Make all data the same shape
    # Build NN to learn what midi should be generated from an audio clip
    # Find a way to compare quantitatively the correct midi against the NN generated midi

    # TODO:
    # Merge all tracks with notes together. metadata track can be kept independent
    # Parse through note-containing tracks (which are lists of messages)
    # Delta time to absolute time n ticks
    # convert Absolute time in ticks to NN standard time (all songs need the same ticks per beat)
    #

    # https://www.youtube.com/watch?v=MhOdbtPhbLU
    # If you end up using pretty_midi in a published research project, please cite the following report:
    #
    # Colin Raffel and Daniel P. W. Ellis. Intuitive Analysis, Creation and Manipulation of MIDI Data with pretty_midi. In 15th International Conference on Music Information Retrieval Late #breaking and Demo Papers, 2014.

