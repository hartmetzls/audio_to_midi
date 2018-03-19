import numpy as np

# midi_start_time = 2.0
# midi_segment = [['note_on', 64, 2.3]]
# midi_segment_length = 0.5

def encode_midi_segment(midi_start_time, midi_segment, midi_segment_length):
    lowest = 21
    highest = 107
    num_notes = highest - lowest + 1
    num_discrete_time_values = 6
    encoded_segment_shape = np.zeros(shape=(num_notes, num_discrete_time_values))
    # for note in range(lowest, highest + 1):
    #     encoded_segment_shape[note]=[]
    for message in midi_segment:
        time = message[-1]
        time_scaled = time - midi_start_time
        pitch = message[1]
        pitch_scaled = pitch - lowest
        bucket_length = midi_segment_length/num_discrete_time_values
        on_or_off = message[0]
        if on_or_off == 'note_on':
            array_value = 1
        if on_or_off == 'note_off':
            array_value = 0
        bucket_count = 0
        added = 0
        while added == 0:
            for next_bucket_start_time in np.arange(bucket_length, midi_segment_length, bucket_length):
                if time_scaled < next_bucket_start_time:
                    encoded_segment_shape[pitch_scaled, bucket_count] = array_value
                    added += 1
                else:
                    bucket_count += 1
    print(encoded_segment_shape)

def main():
    encode_midi_segment(2.0, [['note_on', 64, 2.3]], 0.5)

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
