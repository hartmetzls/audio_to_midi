


def decode_midi_segment(encoded_segment, midi_segment_length, num_discrete_time_values, lowest, highest, num_notes):
    decoded_midi_segment = []
    scaled_notes_to_check_for_off = []
    for i in range(num_discrete_time_values): #for each column
        column = encoded_segment[:,i]
        len_discrete_time_column_in_seconds = midi_segment_length / num_discrete_time_values
        column_start_time = i * len_discrete_time_column_in_seconds
        scaled_note_num = 0
        midi_message = []
        for scaled_note in scaled_notes_to_check_for_off:
            if column[scaled_note] == 0:
                midi_message.append('note_off')
                pitch = scaled_note_num + lowest
                midi_message.append(pitch)
                midi_message.append(column_start_time)
                scaled_notes_to_check_for_off.remove(scaled_note)
        for scaled_note in column:
            if scaled_note == 1 and scaled_note not in scaled_notes_to_check_for_off:
                midi_message.append('note_on')
                pitch = scaled_note_num + lowest
                midi_message.append(pitch)
                midi_message.append(column_start_time)
                scaled_notes_to_check_for_off.append(scaled_note_num)
            scaled_note_num += 1
        if len(midi_message) > 0:
            decoded_midi_segment.append(midi_message)
    return decoded_midi_segment
