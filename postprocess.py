import numpy as np
import pretty_midi

def model_to_midi(generated_indices, note_tokenizer, midi_file_name="result.midi", fs=5, seq_len=50):
    """
    Convert the model generated array to midi object.

    Arguments:
    ==========
    - generated_indices:
        numpy-array of shape (Ty, n_values), matrix of one-hot vectors
        representing the values generated
    - note_tokenizer:
        tokenizer instance
    - midi_file_name:
        file name to be used for the saved midi
    - fs: int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    - seq_len: int
        The sequence length of the music to be input of neural network
    """
    note_str = []
    for i in generated_indices.reshape(-1):
        note_str.append(note_tokenizer.index_to_notes[i])

    array_piano_roll = np.zeros((128, seq_len))
    for index, note in enumerate(note_str):
        if note == 'e':
            pass
        else:
            splitted_note = note.split(',')
            for j in splitted_note:
                array_piano_roll[int(j), index] = 1
    generate_to_midi = piano_roll_to_pretty_midi(array_piano_roll, fs=fs)
    # print("Tempo {}".format(generate_to_midi.estimate_tempo()))
    for note in generate_to_midi.instruments[0].notes:
        note.velocity = 100
    generate_to_midi.write(midi_file_name)
    return generate_to_midi.estimate_tempo()


def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.

    Arguments:
    ==========
    - piano_roll : np.ndarray
        shape=(128,frames), dtype=int
        Piano roll of one instrument
    - fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    - program : int
        The program number of the instrument.

    Returns:
    ========
    - midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm


# def mid2wav(file, tempo):
#     filename = file.split('.midi')[0]
#     mid = MidiFile(file)
#     output = AudioSegment.silent(mid.length * 1000.0)

#     # tempo = 130  # bpm

#     for track in mid.tracks:
#         # position of rendering in ms
#         current_pos = 0.0
#         current_notes = defaultdict(dict)

#         for msg in track:
#             current_pos += ticks_to_ms(msg.time, tempo, mid)
#             if msg.type == 'note_on':
#                 if msg.note in current_notes[msg.channel]:
#                     current_notes[msg.channel][msg.note].append(
#                         (current_pos, msg))
#                 else:
#                     current_notes[msg.channel][msg.note] = [(current_pos, msg)]

#             if msg.type == 'note_off':
#                 start_pos, start_msg = current_notes[msg.channel][msg.note].pop(
#                 )

#                 duration = math.ceil(current_pos - start_pos)
#                 signal_generator = Sine(note_to_freq(msg.note, 500))
#                 # print(duration)
#                 rendered = signal_generator.to_audio_segment(
#                     duration=duration - 50, volume=-20).fade_out(100).fade_in(30)

#                 output = output.overlay(rendered, start_pos)

#     output.export('./' + filename + '.wav', format="wav")


# def ticks_to_ms(ticks, tempo, mid):
#     tick_ms = math.ceil((60000.0 / tempo) / mid.ticks_per_beat)
#     return ticks * tick_ms
