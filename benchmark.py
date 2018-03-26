import random

def benchmark(decoded_midi_start_times_and_segments, index):
    random_start_time_and_segment = random.choice(decoded_midi_start_times_and_segments)
    random_segment = random_start_time_and_segment[1]
    return random_segment