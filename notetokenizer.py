import numpy as np

class NoteTokenizer:

    def __init__(self):
        self.notes_to_index = {}
        self.index_to_notes = {}
        self.num_of_word = 0
        self.unique_word = 0
        self.notes_freq = {}

    def transform(self, list_array):
        """ Transform a list of note in string into index.

        Arguments:
        ==========
        - list_array : list
            list of note in string format

        Returns:
        =======
        - transformed_list: list
            The transformed list in numpy array.

        """
        transformed_list = np.zeros(list_array.shape)
        for i_instance, instance in enumerate(list_array):
            for i_note, note in enumerate(instance):
                transformed_list[i_instance][i_note] = self.notes_to_index[note]

        return transformed_list

    def partial_fit(self, notes):
        """ Partial fit on the dictionary of the tokenizer

        Arguments:
        ==========
        - notes : list
            list of notes

        """
        for note_str in notes:
            # note_str = ','.join(str(a) for a in note)
            if note_str in self.notes_freq:
                self.notes_freq[note_str] += 1
                self.num_of_word += 1
            else:
                self.notes_freq[note_str] = 1
                self.unique_word += 1
                self.num_of_word += 1
                self.notes_to_index[note_str], self.index_to_notes[self.unique_word] = self.unique_word, note_str

    def add_new_note(self, note):
        """ Add a new note into the dictionary

        Arguments:
        ==========
        - note : str
            a new note which is not in dictionary.

        """
        assert note not in self.notes_to_index
        self.unique_word += 1
        self.notes_to_index[note], self.index_to_notes[self.unique_word] = self.unique_word, note
