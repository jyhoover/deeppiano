import pretty_midi
from tqdm import tqdm_notebook
import numpy as np
import pick_midi
from notetokenizer import NoteTokenizer
import random
import tensorflow as tf


def generate_samples(list_all_midi,
                     batch_music=16,
                     fs=5,
                     seq_len=50,
                     use_tqdm=False,
                     random_song=True,
                     random_trim=True):
    """
    Generate Batch music that will be used to be input and output of the neural network

    Arguments:
    ==========
    - list_all_midi : list
        List of midi files' paths
    - batch_music : int
        A number of music in one batch
    - fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    - seq_len : int
        The sequence length of the music to be input of neural network
    - use_tqdm : bool
        Whether to use tqdm or not in the function
    - random_song: bool
        If randomly pick musics from the picked composer's midi list
    - random_trim: bool
        If randomly cut a piece from the picked music

    Returns:
    =======
    Tuple of input and target neural network

    """

    assert len(list_all_midi) >= batch_music
    dict_time_notes = generate_dict_time_notes(
        list_all_midi, batch_music, fs, random_song=random_song)

    list_of_dict_keys_time = process_notes_in_song(dict_time_notes)

    list_of_dict_time_notestr = note_value_to_string(list_of_dict_keys_time)

    X, Y, note_tokenizer, indices_one_hot, list_of_dict_time_notestr_trimmed = prepare_samples(
        list_of_dict_time_notestr, seq_len=seq_len, random_trim=random_trim)

    return X, Y, note_tokenizer, indices_one_hot, list_of_dict_time_notestr_trimmed


def generate_dict_time_notes(list_all_midi, batch_music=16, fs=5, random_song=True):
    """ Generate map (dictionary) of music ( in index ) to piano_roll (in np.array)

    Arguments:
    ==========
    - list_all_midi : list
        List of midi files' paths
    - batch_music : int
        A number of music in one batch
    - fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    - random_song: bool
        If randomly pick musics from the picked composer's midi list

    Returns
    =======
    - dict_time_notes:
        dictionary of music to piano_roll (in np.array),
        1 doctionary with batch_music (16) elements:
        keys (0,1,2,...) vs
        values (binary 2D numpy.array in (notes (128), time) dimention)

    """
    assert len(list_all_midi) >= batch_music
    np.random.seed(42)
    dict_time_notes = {}
    # process_tqdm_midi = tqdm_notebook(range(start_index, min(start_index + batch_music, len(
    #     list_all_midi)))) if use_tqdm else range(start_index,  min(start_index + batch_music, len(list_all_midi)))
    if random_song:
        process_tqdm_midi = np.random.choice(
            len(list_all_midi), batch_music, replace=False)
    else:
        process_tqdm_midi = np.arange(batch_music)
    for i in process_tqdm_midi:
        midi_file_name = list_all_midi[i]
        # if use_tqdm:
        #     process_tqdm_midi.set_description(
        #         "Processing {}".format(midi_file_name))
        try:  # Handle exception on malformat MIDI files
            midi_pretty_format = pretty_midi.PrettyMIDI(midi_file_name)
            # Get the piano channels
            piano_midi = midi_pretty_format.instruments[0]
            piano_roll = piano_midi.get_piano_roll(fs=fs)
            dict_time_notes[i] = piano_roll
        except Exception as e:
            print(e)
            print("broken file : {}".format(midi_file_name))
            pass
    return dict_time_notes


def process_notes_in_song(dict_time_notes):
    """
    Iterate the dict of piano rolls into dictionary of timesteps and note played

    Arguments:
    ==========
    - dict_time_notes:
        dictionary of music to piano_roll (in np.array),
        1 doctionary with batch_music (16) elements:
        keys (0,1,2,...) vs.
        values (binary 2D numpy.array in (notes (128), time) dimention)

    Returns
    =======
    - list_of_dict_keys_time: list
        List of dict of timesteps and note played
        A list with len of batch_song (16), each element is a dict
        Each dict has keys (timestep numbers: 30,31,35,...) vs.
        values (note arrays, e.g. array([36, 48, 60])
        such as {5: array([36, 48, 60]),
                 6: array([36, 48, 60]),
                 7: array([36, 39, 48, 51, 60, 63]),
                 12: array([51, 63, 75]),
                 17: array([79]),...}
    """
    list_of_dict_keys_time = []

    for key in dict_time_notes:
        sample = dict_time_notes[key]
        times = np.unique(np.where(sample > 0)[1])
        index = np.where(sample > 0)
        dict_keys_time = {}

        for time in times:
            index_where = np.where(index[1] == time)
            notes = index[0][index_where]
            dict_keys_time[time] = notes
        list_of_dict_keys_time.append(dict_keys_time)
    return list_of_dict_keys_time


def tokenize_note(list_of_dict_time_notestr):
    """
    Arguments:
    ==========
    - list_of_dict_keys_time: list
        List of dict of timesteps and note played
        A list with len of batch_song (16), each element is a dict
        Each dict has keys (timestep numbers: 30,31,35,...) vs.
        values (note arrays, e.g. array([36, 48, 60])
        such as {5: array([36, 48, 60]),
                 6: array([36, 48, 60]),
                 7: array([36, 39, 48, 51, 60, 63]),
                 12: array([51, 63, 75]),
                 17: array([79]),...}

    Returns
    =======
    - note_tokenizer: NoteTokenizer instance
      Attributes:
        notes_to_index: dict
        index_to_notes: dict
        num_of_word: int
        unique_word: int
        notes_freq: dict
    """
    note_tokenizer = NoteTokenizer()
    for song in list_of_dict_keys_time:
        note_tokenizer.partial_fit(list(song.values()))
    return note_tokenizer


def note_value_to_string(list_of_dict_keys_time):
    """
    Convert note arrays (e.g. array([36, 48, 60]) to strings (e.g. '36,48,60')

    Arguments:
    ==========
    - list_of_dict_keys_time : list
        List of dict of timesteps and note played
        A list with len of batch_song (16), each element is a dict
        Each dict has keys (timestep numbers: 30,31,35,...) vs.
        values (note arrays, e.g. array([36, 48, 60])


    Returns
    =======
    - list_of_dict_time_notestr: list
        List of dict of timesteps and note played
        A list with len of batch_song (16), each element is a dict
        Each dict has keys (timestep numbers: 30,31,35,...) vs.
        values (note strings, e.g. '36,48,60')
    """
    list_of_dict_time_notestr = []
    for song_dict in list_of_dict_keys_time:
        dict_time_notestr = {}
        for time in song_dict:
            note_str = ','.join(str(a) for a in song_dict[time])
            dict_time_notestr[time] = note_str
        list_of_dict_time_notestr.append(dict_time_notestr)
    return list_of_dict_time_notestr


def prepare_samples(list_of_dict_time_notestr, seq_len=50, random_trim=True):
    """
    Arguments:
    ==========
    - list_of_dict_time_notestr: list
        List of dict of timesteps and note played
        A list with len of batch_song (16), each element is a dict
        Each dict has keys (timestep numbers: 30,31,35,...) vs.
        values (note strings, e.g. '36,48,60')
    - seq_len: int
        The sequence length of the music to be input of neural network
    - random_trim: bool
        If randomly cut a piece from the picked music

    Returns
    =======
    - list_of_dict_time_notestr_trimmed: list
          size(m,Tx,1)
          m: samples, batch_song (=16)
          Tx: sequence length, seq_len (=50)
          contains note strings
    - X:
    - Y:
    - note_tokenizer:
    - indices_one_hot:
    """
    random.seed(42)
    list_of_dict_time_notestr_trimmed = np.empty(
        (len(list_of_dict_time_notestr), seq_len), dtype=np.dtype((np.unicode_, 64)))
    for i_enum, song_dict in enumerate(list_of_dict_time_notestr):
        assert len(song_dict) >= seq_len
        # randomly pick a start timestep
        if random_trim:
            start_point = random.choice(list(range(len(song_dict) - seq_len)))
        else:
            start_point = 0
        start_time = list(song_dict.keys())[0] + start_point
        while start_time not in list(song_dict.keys()):
            start_time -= 1

        for i_seq_len in range(seq_len):
            if start_time + i_seq_len in list(song_dict.keys()):
                list_of_dict_time_notestr_trimmed[i_enum][i_seq_len] = song_dict[start_time + i_seq_len]
            else:
                list_of_dict_time_notestr_trimmed[i_enum][i_seq_len] = 'e'
    note_tokenizer = NoteTokenizer()
    note_tokenizer.partial_fit(list_of_dict_time_notestr_trimmed.reshape(-1))
    n_unique = note_tokenizer.unique_word
    indices_one_hot = note_tokenizer.transform(
        list_of_dict_time_notestr_trimmed)
    X = tf.one_hot(indices_one_hot, n_unique)
    Y = np.roll(X, -1, axis=1)
    Y = Y.transpose(1, 0, 2)
    return X, Y, note_tokenizer, indices_one_hot, list_of_dict_time_notestr_trimmed
