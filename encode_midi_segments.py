import numpy as np

def encode_midi_segment(midi_start_time, midi_segment, midi_segment_length, lowest, highest):
    lowest = 21 #value for the entire dataset
    highest = 107 #value for the entire dataset
    num_notes = highest - lowest + 1
    num_discrete_time_values = 6
    #instantiate shape
    encoded_segment = np.zeros(shape=(num_notes, num_discrete_time_values))
    midi_segment_bin_time = []
    times_aligned_to_closest_bucket_divide = []
    bucket_length = midi_segment_length / num_discrete_time_values
    bucket_divides = np.arange(0, midi_segment_length, bucket_length)
    for message in midi_segment:
        time = message[-1]
        time_scaled = time - midi_start_time
        time_aligned_to_closest_bucket_divide = min(bucket_divides, key=lambda x:abs(x-time_scaled))
        times_aligned_to_closest_bucket_divide.append(time_aligned_to_closest_bucket_divide)
    bins = []
    for time in times_aligned_to_closest_bucket_divide:
        nth_bucket = 0
        for bucket in bucket_divides:
            if time == bucket:
                bins.append(nth_bucket)
            nth_bucket += 1
    i = 0
    for message in midi_segment:
        on_or_off = message[0]
        pitch = message[1]
        pitch_scaled = pitch - lowest
        bin = bins[i]
        midi_segment_bin_time.append([on_or_off, pitch_scaled, bin])
        i += 1

    # on = []
    for message in midi_segment_bin_time:
        on_or_off = message[0]
        pitch_scaled = message[1]
        bin = message[-1]
        if on_or_off == 'note_on':
            encoded_segment[pitch_scaled, bin:] = 1
            # on.append(pitch)
        else:  # if note off
            if encoded_segment[pitch_scaled, bin - 1] == 1: #if note was not turned on inside this bin
                encoded_segment[pitch_scaled, bin:] = 0
            # on.remove(pitch)

    return encoded_segment, num_discrete_time_values, num_notes

    #         #TODO: Notes which are shorter than .08, and don't happen to cross a bucket end time line, are completely left out. Should the solution be to represent them as longer than they should be OR leave them out completely? Maybe this decision can be based on how long, for ex, leave out notes shorter than .04 and leave in those longer than .04. ANSWER: Every note present in the segment should be represented in at least one bucket.
    #         # For
    #         # notes that span multiple buckets, the end buckets should be rounded and if they less
    #         # than half fill the end bucket, do not include
    #
    #         e



# def incorporate_msg_into_encoded_segment(bucket_length, encoded_segment, midi_segment_length,
#                                          num_discrete_time_values, on_or_off, pitch_scaled,
#                                          time_scaled):
#     if on_or_off == 'note_on':
#         bucket_count = 0
#         for next_bucket_start_time in np.arange(bucket_length, midi_segment_length,
#                                                 bucket_length):
#             if time_scaled < next_bucket_start_time:
#                 encoded_segment[pitch_scaled, bucket_count] = 1  # TODO: Is there better way to
#                 #  get this index than bucket count?
#                 return
#             bucket_count += 1
#     else:  # if note off
#         bucket_count = 0
#         for next_bucket_start_time in np.arange(bucket_length, midi_segment_length,
#                                                 bucket_length):
#             last_bucket = num_discrete_time_values - 1
#             if time_scaled < next_bucket_start_time:
#                 if encoded_segment[pitch_scaled, bucket_count] == 1:
#                     if bucket_count != last_bucket:
#                         encoded_segment[
#                             pitch_scaled, bucket_count + 1] = 1  # turn off in next bucket
#                         return
#                 else:
#                     encoded_segment[pitch_scaled, bucket_count] = 0
#                     return
#             bucket_count += 1

#     #a = numpy.zeros(shape=(5,2))
# >>> a
# array([[ 0.,  0.],
#    [ 0.,  0.],
#    [ 0.,  0.],
#    [ 0.,  0.],
#    [ 0.,  0.]])
# >>> a[0] = [1,2]
# >>> a[1] = [2,3]
# >>> a
# array([[ 1.,  2.],
#    [ 2.,  3.],
#    [ 0.,  0.],
#    [ 0.,  0.],
#    [ 0.,  0.]])
