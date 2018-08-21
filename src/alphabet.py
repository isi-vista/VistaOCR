class Alphabet(object):

    def __init__(self, char_array, left_to_right=False):

        self.left_to_right = left_to_right
        self.char_to_idx = dict(zip(char_array, range(len(char_array))))
        self.idx_to_char = dict(zip(range(len(char_array)), char_array))
        self.char_array = char_array


    def __len__(self):
        return len(self.idx_to_char)
