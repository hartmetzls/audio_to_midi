import numpy as np

def encode_midi_segment(midi_start_time, midi_segment, midi_segment_length, lowest, highest):
    lowest = 21 #value for the entire SMD dataset
    highest = 107 #value for the entire SMD dataset
    num_notes = highest - lowest + 1
    num_discrete_time_values = 6
    # instantiate shape
    encoded_segment = np.zeros(shape=(num_notes, num_discrete_time_values))
    midi_segment_bin_time = []
    times_aligned_to_closest_bucket_divide = []
    bucket_length = midi_segment_length / num_discrete_time_values
    bucket_divides = np.arange(0, midi_segment_length, bucket_length)

    for message in midi_segment:
        time = message[-1]
        time_scaled = time - midi_start_time
        time_aligned_to_closest_bucket_divide = min(bucket_divides, key=lambda x:abs(x-time_scaled))
        # add the time to the list of message times we'll have to handle
        times_aligned_to_closest_bucket_divide.append(time_aligned_to_closest_bucket_divide)

    # create a list of bin integers (corresponding to each message in the segment)
    bins = []
    for time in times_aligned_to_closest_bucket_divide:
        nth_bucket = 0
        for bucket in bucket_divides:
            if time == bucket:
                bins.append(nth_bucket)
            nth_bucket += 1

    # build midi messages list with bin integers as time value
    i = 0
    for message in midi_segment:
        on_or_off = message[0]
        pitch = message[1]
        pitch_scaled = pitch - lowest
        bin = bins[i]
        midi_segment_bin_time.append([on_or_off, pitch_scaled, bin])
        i += 1

    for message in midi_segment_bin_time:
        on_or_off = message[0]
        pitch_scaled = message[1]
        bin = message[-1]
        if on_or_off == 'note_on':
            encoded_segment[pitch_scaled, bin:] = 1 # turn note on for the rest of the segment
        else:  # if note off
            if encoded_segment[pitch_scaled, bin - 1] == 1: # if note was not turned on inside this bin
                encoded_segment[pitch_scaled, bin:] = 0

    return encoded_segment, num_discrete_time_values, num_notes

