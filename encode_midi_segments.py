import numpy as np

# midi_start_time = 2.0
# midi_segment = [['note_on', 64, 2.3]]
# midi_segment_length = 0.5

def encode_midi_segment(midi_start_time, midi_segment, midi_segment_length):
    lowest = 21
    highest = 107
    num_notes = highest - lowest + 1
    num_discrete_time_values = 6
    #instantiate shape
    encoded_segment = np.zeros(shape=(num_notes, num_discrete_time_values))
    # for note in range(lowest, highest + 1):
    #     encoded_segment[note]=[]
    for message in midi_segment:
        time = message[-1]
        time_scaled = time - midi_start_time
        bucket_length = midi_segment_length / num_discrete_time_values
        pitch = message[1]
        pitch_scaled = pitch - lowest
        on_or_off = message[0]
        if on_or_off == 'note_on':
            array_value = 1
        else:
            array_value = 0
        bucket_count = 0
        for next_bucket_start_time in np.arange(bucket_length, midi_segment_length, bucket_length):
            if time_scaled < next_bucket_start_time:
                encoded_segment[pitch_scaled, bucket_count] = array_value

            #TODO: Notes which are shorter than .08, and don't happen to cross a bucket end time line, are completely left out. Should the solution be to represent them as longer than they should be OR leave them out completely? Maybe this decision can be based on how long, for ex, leave out notes shorter than .04 and leave in those longer than .04

            #this condition is here for 'note off's which were added in in the course of ensuring a note off for every note on.
            elif time_scaled == midi_segment_length:
                last_bucket = num_discrete_time_values - 1
                encoded_segment[pitch_scaled, last_bucket] = array_value
            bucket_count += 1
    return encoded_segment

def main():
    #exploratory exampled
    encode_midi_segment(2.0, [['note_on', 64, 2.3], ['note_off', 64, 2.49]], 0.5)

if __name__ == '__main__':
    main()

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
