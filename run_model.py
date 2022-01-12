import numpy as np
from fractions import Fraction
import pick_midi
import preprocess
import postprocess
import deep_piano_model
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


def main_run(random_song, random_trim, composer_num,
                fs, seq_len, batch_music, n_a, epochs,
                opt, print_note=False):
    """
    'main_run' integrates all major precedures.

    Arguments:
    ==========
    - random_song: bool
        If randomly pick musics from the picked composer's midi list
    - random_trim: bool
        If randomly cut a piece from the picked music
    - composer_num: int
        The code to designate a composer
    - fs: int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    - seq_len: int
        The sequence length of the music to be input of neural network
    - batch_music: int
        A number of music in one batch
    - n_a: int
        The number of the hidden state vector
    - epochs: int
        The number of how many times the model will be trained
    - opt: optimizer
        a tensorflow optimizer
    - print_note: bool
        If print the music notes. Only for the purposes of testing


    Returns:
    ========
    None

    Generated midi files will be saved under the 'result' folder
    """
    # pick the midi list, store in a dict
    composer_dict = composer_list()
    # obtain the list of the path of the piano pieces of the chosen composor
    list_all_midi = pick_midi.pick_file(composer_dict,
                                        composer_num=composer_num)
    # generate the samples
    X, Y, note_tokenizer, indices_one_hot, list_of_dict_time_notestr_trimmed =\
    preprocess.generate_samples(list_all_midi,batch_music=batch_music,fs=fs,
    seq_len=seq_len,use_tqdm=False,random_song=random_song,random_trim=random_trim)
    # file name
    filename1 = '''/content/drive/MyDrive/Colab_Notebooks/deeppiano/result/result_'''
    filename2 = '''composer({})-fs({})-seq_len({})-batch_music({})-random_song({})
                    -random_trim({})-na({})-epochs({})-'''.format(
                                            composer_dict[int(composer_num)],
                                            fs, seq_len, batch_music, random_song,
                                            random_trim, n_a, epochs)
    filename3 = '-'.join([f"{key}({str(val).replace('.', '_')})" \
          if isinstance(val, float) else f"{key}({str(val)})" \
          for key,val in opt.get_config().items()])
    file = filename1 + filename2 + filename3
    if print_note:
        print('index_to_notes:\n', note_tokenizer.index_to_notes)
    print('X shape: {}\nY shape:{}'.format(X.shape, Y.shape))
    print('file name:\n', file)
    # set the training parameters and shared layers
    Tx = seq_len
    Ty = Tx
    n_values = note_tokenizer.unique_word
    reshaper = tfl.Reshape((1, n_values))
    LSTM_cell = tfl.LSTM(n_a, return_state=True)
    densor = tfl.Dense(n_values, activation='softmax')
    # build and fit the model
    model = deep_piano_model.learning_model(
        Tx=Tx, LSTM_cell=LSTM_cell, densor=densor, reshaper=reshaper)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    a0 = np.zeros((batch_music, n_a))
    c0 = np.zeros((batch_music, n_a))
    history = model.fit([X, a0, c0], list(Y), epochs=epochs, verbose=0)
    # save the loss function
    print(f"loss at epoch 1: {history.history['loss'][0]}")
    print(f"loss at epoch {epochs}: {history.history['loss'][epochs-1]}")
    plt.plot(history.history['loss'],
             label=f'''random_song:{random_song}; random_trim:{random_trim};
                         fs:{fs}; seq_len:{seq_len}; batch_music:{batch_music};
                         n_a:{n_a}; epochs:{epochs}''')
    plt.legend(loc='upper center', bbox_to_anchor=(2, 1))
    plt.savefig(file, bbox_inches="tight")
    # inference model, prediction and sampling
    inference_model = deep_piano_model.music_inference_model(
        LSTM_cell, densor, Ty=Ty)
    x_initializer = np.zeros((1, 1, n_values))
    a_initializer = np.zeros((1, n_a))
    c_initializer = np.zeros((1, n_a))
    generated_results, generated_indices = deep_piano_model.predict_and_sample(
        inference_model, x_initializer, a_initializer, c_initializer)
    print('generated_results.shape: ', generated_results.shape)
    if print_note:
        print('generated_indices:\n', generated_indices)
    # index_to_notes starts from 1 (1,...,n_values), but one hot indices start from 0 (0,...,n_values-1)
    # change 0 in one hot indices to n_values for index_to_notes
    with np.nditer(generated_indices, op_flags=['readwrite']) as it:
        for x in it:
            if x == 0:
                x[...] = n_values
    postprocess.model_to_midi(generated_indices, note_tokenizer,
                              midi_file_name=file + '.mid', fs=fs, seq_len=seq_len)


def composer_list():
    """
    Returns a dictionary of {code: composer} pairs
    """
    composer_dict = \
        {0: 'Albéniz', 1: 'Bach', 2: 'Balakirev', 3: 'Beethoven', 4: 'Berg',
         5: 'Bizet', 6: 'Brahms', 7: 'Busoni', 8: 'Chopin', 9: 'Clementi',
         10: 'Cziffra', 11: 'Debussy', 12: 'Enescu', 13: 'Fischer', 14: 'Franck',
         15: 'Gibbons', 16: 'Glinka', 17: 'Godowsky', 18: 'Gounod', 19: 'Grainger',
         20: 'Grieg', 21: 'Gryaznov', 22: 'Grünfeld', 23: 'Handel', 24: 'Haydn',
         25: 'Hess', 26: 'Horowitz', 27: 'Janáček', 28: 'Kreisler', 29: 'Liszt',
         30: 'Medtner', 31: 'Mendelssohn', 32: 'Moszkowski', 33: 'Mozart', 34: 'Mussorgsky',
         35: 'Pachelbel', 36: 'Paganini', 37: 'Petri', 38: 'Pletnev', 39: 'Purcell',
         40: 'Rachmaninoff', 41: 'Rameau', 42: 'Rimsky-Korsakov', 43: 'Saint-Saëns', 44: 'Scarlatti',
         45: 'Schubert', 46: 'Schumann', 47: 'Scriabin', 48: 'Soler', 49: 'Strauss',
         50: 'Tchaikovsky', 51: 'Verdi', 52: 'Wagner', 53: 'von Weber'}
    return composer_dict
