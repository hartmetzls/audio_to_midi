


def decode_midi_segment(encoded_segment, midi_segment_length, num_discrete_time_values, lowest, highest, num_notes):
    for i in range(num_discrete_time_values): #iter through columns
        column = encoded_segment[:,i]
        scaled_note_num = 0
        for note in column:
            if note == 1:
                midi_message = ['note_on']
                index_note = column[note]
    print('butts. that is all.')
    # return decoded_midi_segment
