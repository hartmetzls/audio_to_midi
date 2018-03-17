import numpy as np

def encode_midi_segment(midi_start_time, midi_segment):
    lowest = 21
    highest = 107
    num_notes = highest - lowest + 1
    num_discrete_time_values = 6
    encoded_segment_shape = np.zeros(shape=(num_notes, num_discrete_time_values))
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
    for message in midi_segment:
        encoded_segment_shape[]