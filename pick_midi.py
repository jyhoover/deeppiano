"""
midi source: https://magenta.tensorflow.org/datasets/maestro
"""
import pandas as pd


def pick_file(composer_dict, composer_num=33):
    """
    pick the midi files of the chosen composer.

    Arguments:
    ==========
    - composer_dict:
        A dictionary of {code: composer} pairs
    - composer_num:
        The code to designate a composer

    Returns:
    ========
    - list_all_midi : list
        List of midi files' paths
    """
    dfmidi = pd.read_csv(
        '/content/drive/MyDrive/Colab_Notebooks/deeppiano/maestro-v3.0.0/maestro-v3.0.0.csv')
    # ['Alban Berg' 'Alexander Scriabin' 'Antonio Soler' 'Carl Maria von Weber'
    # 'Charles Gounod / Franz Liszt' 'Claude Debussy' 'César Franck'
    # 'Domenico Scarlatti' 'Edvard Grieg' 'Felix Mendelssohn'
    # 'Felix Mendelssohn / Sergei Rachmaninoff' 'Franz Liszt'
    # 'Franz Liszt / Camille Saint-Saëns' 'Franz Liszt / Vladimir Horowitz'
    # 'Franz Schubert' 'Franz Schubert / Franz Liszt'
    # 'Franz Schubert / Leopold Godowsky'
    # 'Fritz Kreisler / Sergei Rachmaninoff' 'Frédéric Chopin' 'George Enescu'
    # 'George Frideric Handel' 'Georges Bizet / Ferruccio Busoni'
    # 'Georges Bizet / Moritz Moszkowski' 'Georges Bizet / Vladimir Horowitz'
    # 'Giuseppe Verdi / Franz Liszt' 'Henry Purcell' 'Isaac Albéniz'
    # 'Isaac Albéniz / Leopold Godowsky' 'Jean-Philippe Rameau'
    # 'Johann Christian Fischer / Wolfgang Amadeus Mozart' 'Johann Pachelbel'
    # 'Johann Sebastian Bach' 'Johann Sebastian Bach / Egon Petri'
    # 'Johann Sebastian Bach / Ferruccio Busoni'
    # 'Johann Sebastian Bach / Franz Liszt' 'Johann Sebastian Bach / Myra Hess'
    # 'Johann Strauss / Alfred Grünfeld' 'Johannes Brahms' 'Joseph Haydn'
    # 'Leoš Janáček' 'Ludwig van Beethoven' 'Mikhail Glinka / Mily Balakirev'
    # 'Mily Balakirev' 'Modest Mussorgsky' 'Muzio Clementi'
    # 'Niccolò Paganini / Franz Liszt' 'Nikolai Medtner'
    # 'Nikolai Rimsky-Korsakov / Sergei Rachmaninoff' 'Orlando Gibbons'
    # 'Percy Grainger' 'Pyotr Ilyich Tchaikovsky'
    # 'Pyotr Ilyich Tchaikovsky / Mikhail Pletnev'
    # 'Pyotr Ilyich Tchaikovsky / Sergei Rachmaninoff'
    # 'Richard Wagner / Franz Liszt' 'Robert Schumann'
    # 'Robert Schumann / Franz Liszt' 'Sergei Rachmaninoff'
    # 'Sergei Rachmaninoff / György Cziffra'
    # 'Sergei Rachmaninoff / Vyacheslav Gryaznov' 'Wolfgang Amadeus Mozart']

    # Index(['canonical_composer', 'canonical_title', 'split', 'year',
    #  'midi_filename', 'audio_filename', 'duration'],
    # dtype='object')

    result = dfmidi[[composer_dict[composer_num]
                     in c for c in dfmidi.canonical_composer]].midi_filename.tolist()
    list_all_midi = [
        '/content/drive/MyDrive/Colab_Notebooks/deeppiano/maestro-v3.0.0/' + r for r in result]
    return list_all_midi
