import os

class EnglishAlphabet(object):
    punctuations = {
        "u002e", "u002c", "u003b", "u0027", "u0022", "u002f", "u0021", "u0028", "u0029", "u005b", "u005d", "u003c",
        "u003e", "u002d", "u005f",
        "u007b", "u007d", "u0024", "u0025", "u0023", "u0026", "u002a"
    }
    digits = {
        "u0030", "u0031", "u0032", "u0033", "u0034", "u0035", "u0036", "u0037", "u0038", "u0039"
    }
    
    def __init__(self, lm_units_path):
        self.name = "English"
        self.left_to_right = True
        self.idx_to_alphabet = ['<ctc-blank>',
                                'u0061', 'u0062', 'u0063', 'u0064', 'u0065', 'u0066', 'u0067', 'u0068', 'u0069',
                                'u006a', 'u006b',
                                'u006c', 'u006d', 'u006e', 'u006f', 'u0070', 'u0071', 'u0072', 'u0073', 'u0074',
                                'u0075', 'u0076',
                                'u0077', 'u0078', 'u0079', 'u007a', 'u0041', 'u0042', 'u0043', 'u0044', 'u0045',
                                'u0046', 'u0047',
                                'u0048', 'u0049', 'u004a', 'u004b', 'u004c', 'u004d', 'u004e', 'u004f', 'u0050',
                                'u0051', 'u0052',
                                'u0053', 'u0054', 'u0055', 'u0056', 'u0057', 'u0058', 'u0059', 'u005a', 'u0030',
                                'u0031', 'u0032',
                                'u0033', 'u0034', 'u0035', 'u0036', 'u0037', 'u0038', 'u0039', 'u0020', 'u002e',
                                'u002c', 'u003b',
                                'u005b', 'u005d', 'u007b', 'u007d', 'u0028', 'u0029', 'u002d', 'u003a', 'u003c',
                                'u003e', 'u003f',
                                'u002f', 'u003d', 'u002a', 'u0026', 'u005e', 'u0025', 'u0024', 'u0023', 'u0040',
                                'u0021', 'u007e',
                                'u0060', 'u002b', 'u002d', 'u005f', 'u007c', 'u0022', 'u0027']

        self.lmidx_to_alphabet = EnglishAlphabet.extract_lm_units(lm_units_path) if lm_units_path is not None else []
        self.alphabet_to_idx = dict()
        for idx, char in enumerate(self.idx_to_alphabet):
            self.alphabet_to_idx[char] = idx

        self.alphabet_to_lmidx = dict()
        for idx, char in enumerate(self.lmidx_to_alphabet):
            self.alphabet_to_lmidx[char] = idx

    @staticmethod
    def extract_lm_units(units_path):
        assert os.path.exists(units_path), 'lm units.txt path not found:%s' % units_path
        units = ['<ctc-blank>']
        with open(units_path, 'r') as fh:
            for line in fh:
                units.append(line.strip().split(' ')[0])
        return units
        

    def __len__(self):
        return len(self.idx_to_alphabet)
