import numpy as np

def decode_midi_segment(encoded_segment, midi_segment_length, num_discrete_time_values, lowest):
    lowest = 21  # value for the entire SMD dataset
    decoded_midi_segment = []

    notes_on, times_on = np.where(encoded_segment == 1) #tuple of arrays of x and y indices where a given condition holds in an array.3.0
    bucket_length = midi_segment_length / num_discrete_time_values
    bucket_start_times = np.arange(0, midi_segment_length, bucket_length)
    for i in range(len(notes_on)):
        pitch_scaled = notes_on[i]
        column = times_on[i]
        pitch = pitch_scaled + lowest
        previous_pitch_scaled = notes_on[i-1]
        if pitch_scaled != previous_pitch_scaled:
            note_on_time = bucket_start_times[column]
            midi_message = ['note_on', pitch, note_on_time]
            decoded_midi_segment.append(midi_message)
        last_column = num_discrete_time_values - 1
        if column != last_column:
            #if not last note in notes on
            if pitch_scaled != notes_on[-1]:
                next_pitch_scaled = notes_on[i+1]
                if pitch_scaled != next_pitch_scaled:
                    note_off_time = bucket_start_times[column + 1]
                    midi_message = ['note_off', pitch, note_off_time]
                    decoded_midi_segment.append(midi_message)
        if column == last_column:
            note_off_time = bucket_start_times[0] + midi_segment_length
            midi_message = ['note_off', pitch, note_off_time]
            decoded_midi_segment.append(midi_message)
    decoded_midi_segment_chronological = sorted(decoded_midi_segment, key=lambda x: float(x[-1]))
    return decoded_midi_segment_chronological

















    # scaled_notes_on = []
    # len_discrete_time_column_in_seconds = midi_segment_length / num_discrete_time_values
    # for i in range(num_discrete_time_values): #for each column
    #     column = encoded_segment[:,i]
    #     column_start_time = i * len_discrete_time_column_in_seconds
    #     scaled_note_num = 0
    #     midi_message = []
    #
    #     for scaled_note in scaled_notes_on:
    #         if column[scaled_note] == 0:
    #             midi_message.append('note_off')
    #             pitch = scaled_note_num + lowest
    #             midi_message.append(pitch)
    #             midi_message.append(column_start_time)
    #             scaled_notes_on.remove(scaled_note)
    #
    #     for binary_note in column: #just one column
    #         if binary_note == 1 and scaled_note_num not in scaled_notes_on:
    #             midi_message.append('note_on')
    #             pitch = scaled_note_num + lowest
    #             midi_message.append(pitch)
    #             midi_message.append(column_start_time)
    #             scaled_notes_on.append(scaled_note_num)
    #         scaled_note_num += 1
    #
    #     if len(midi_message) > 0:
    #         decoded_midi_segment.append(midi_message)
    return decoded_midi_segment_bin_time
