import os

class ArabicAlphabet(object):
#    punctuations = {
#        "u002e", "u002c", "u003b", "u0027", "u0022", "u002f", "u0021", "u0028", "u0029", "u005b", "u005d", "u003c", "u003e",
#        "u002d", "u005f", "u007b", "u007d", "u0024", "u0025", "u0023", "u0026", "u060c", "u201d", "u060d", "u060f", "u061f",
#        "u066d", "ufd3e", "ufd3f", "u061e", "u066a", "u066b", "u066c"
#    }

    punctuations = {"u002e", "u002c", "u003b", "u0027", "u0022", "u002f", "u0021", "u0028", "u0029", "u005b", "u005d", "u003c", "u003e",
                   "u002d", "u005f", "u007b", "u007d", "u0024", "u0025", "u0023", "u0026", "u060c", "u201d", "u060d", "u060f", "u061f",
                   "u066d", "ufd3e", "ufd3f", "u061e", "u066a", "u066b", "u066c", "u002a", "u002b", "u003a", "u003d", "u005e", "u0060", "u007c", "u007e"}


    digits = {
        "u0660", "u0661", "u0662", "u0663", "u0664", "u0665", "u0666", "u0667", "u0668", "u0669", "u0030", "u0031",
        "u0032", "u0033", "u0034", "u0035", "u0036", "u0037", "u0038", "u0039"
    }
        
    def __init__(self, lm_units_path=None):
        self.name = "Arabic"
        self.left_to_right = False
        self.idx_to_alphabet = ['<ctc-blank>',
                                'u0020', 'u0021', 'u0022', 'u0024', 'u0025', 'u0026', 'u0027', 'u0028', 'u0029',
                                'u002a', 'u002b',
                                'u002c', 'u002d', 'u002e', 'u002f', 'u0030', 'u0031', 'u0032', 'u0033', 'u0034',
                                'u0035', 'u0036',
                                'u0037', 'u0038', 'u0039', 'u003a', 'u003b', 'u003c', 'u003d', 'u003e', 'u003f',
                                'u0040', 'u0041',
                                'u0042', 'u0043', 'u0044', 'u0045', 'u0046', 'u0047', 'u0048', 'u0049', 'u004a',
                                'u004b', 'u004c',
                                'u004d', 'u004e', 'u004f', 'u0050', 'u0051', 'u0052', 'u0053', 'u0054', 'u0055',
                                'u0056', 'u0057',
                                'u0058', 'u005a', 'u005b', 'u005c', 'u005d', 'u005e', 'u005f', 'u0061', 'u0062',
                                'u0063', 'u0064',
                                'u0065', 'u0066', 'u0067', 'u0068', 'u0069', 'u006a', 'u006b', 'u006c', 'u006d',
                                'u006e', 'u006f',
                                'u0070', 'u0071', 'u0072', 'u0073', 'u0074', 'u0075', 'u0076', 'u0077', 'u0078',
                                'u0079', 'u007a',
                                'u007b', 'u007d', 'u007e', 'u00a9', 'u00ab', 'u00ad', 'u00b7', 'u00bb', 'u00be',
                                'u00d7', 'u00e8',
                                'u00ec', 'u060c', 'u061b', 'u061f', 'u0621', 'u0622', 'u0623', 'u0624', 'u0625',
                                'u0626', 'u0627',
                                'u0628', 'u0629', 'u062a', 'u062b', 'u062c', 'u062d', 'u062e', 'u062f', 'u0630',
                                'u0631', 'u0632',
                                'u0633', 'u0634', 'u0635', 'u0636', 'u0637', 'u0638', 'u0639', 'u063a', 'u0640',
                                'u0641', 'u0642',
                                'u0643', 'u0644', 'u0645', 'u0646', 'u0647', 'u0648', 'u0649', 'u064a', 'u064b',
                                'u064c', 'u064d',
                                'u064e', 'u064f', 'u0650', 'u0651', 'u0652', 'u066a', 'u06d2', 'u200c', 'u200d',
                                'u200e', 'u200f',
                                'u2013', 'u2014', 'u2018', 'u2019', 'u201c', 'u201d', 'u2022', 'u2026', 'u25cf',
                                'ufe87', 'ufef9']

        self.lmidx_to_alphabet = ArabicAlphabet.extract_lm_units(lm_units_path) if lm_units_path is not None else []
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
