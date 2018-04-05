import random
import pickle

def benchmark():
    with open('cqt_segments_midi_segments.pkl', 'rb') as handle:
        pickle.load(handle)
        cqt_segments, midi_segments = pickle.load(handle)

    for cqt_segment in cqt_segments:


    num_data_points = len(midi_segments)
    random_index = random.choice(num_data_points)
    random_midi_segment = midi_segments[random_index]
    return random_segment

    benchmark
    for cqt_segment in cqt_segments:
        index = cqt_segments.index(cqt_segment)
        random_segment = benchmark(decoded_midi_start_times_and_segments, index)
        return


    # with open('cqt_segments_midi_segments.pkl', 'wb') as handle:
    #     pickle.dump(cqt_segments, handle)
    #     pickle.dump(all_songs_encoded_midi_segments, handle)

# TODO: Random seed reg, np,