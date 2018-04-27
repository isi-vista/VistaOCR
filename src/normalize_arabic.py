import sys


input_file = sys.argv[1]
output_file = sys.argv[2]

output_lines = []

BASE_TO_PRESENTATION = {
    # ARABIC LETTER HAMZA
    'u0621': ('uFE80', '', '', ''),
    # ARABIC LETTER ALEF WITH MADDA ABOVE
    'u0622': ('uFE81', '', '', 'uFE82'),
    # ARABIC LETTER ALEF WITH HAMZA ABOVE
    'u0623': ('uFE83', '', '', 'uFE84'),
    # ARABIC LETTER WAW WITH HAMZA ABOVE
    'u0624': ('uFE85', '', '', 'uFE86'),
    # ARABIC LETTER ALEF WITH HAMZA BELOW
    'u0625': ('uFE87', '', '', 'uFE88'),
    # ARABIC LETTER YEH WITH HAMZA ABOVE
    'u0626': ('uFE89', 'uFE8B', 'uFE8C', 'uFE8A'),
    # ARABIC LETTER ALEF
    'u0627': ('uFE8D', '', '', 'uFE8E'),
    # ARABIC LETTER BEH
    'u0628': ('uFE8F', 'uFE91', 'uFE92', 'uFE90'),
    # ARABIC LETTER TEH MARBUTA
    'u0629': ('uFE93', '', '', 'uFE94'),
    # ARABIC LETTER TEH
    'u062A': ('uFE95', 'uFE97', 'uFE98', 'uFE96'),
    # ARABIC LETTER THEH
    'u062B': ('uFE99', 'uFE9B', 'uFE9C', 'uFE9A'),
    # ARABIC LETTER JEEM
    'u062C': ('uFE9D', 'uFE9F', 'uFEA0', 'uFE9E'),
    # ARABIC LETTER HAH
    'u062D': ('uFEA1', 'uFEA3', 'uFEA4', 'uFEA2'),
    # ARABIC LETTER KHAH
    'u062E': ('uFEA5', 'uFEA7', 'uFEA8', 'uFEA6'),
    # ARABIC LETTER DAL
    'u062F': ('uFEA9', '', '', 'uFEAA'),
    # ARABIC LETTER THAL
    'u0630': ('uFEAB', '', '', 'uFEAC'),
    # ARABIC LETTER REH
    'u0631': ('uFEAD', '', '', 'uFEAE'),
    # ARABIC LETTER ZAIN
    'u0632': ('uFEAF', '', '', 'uFEB0'),
    # ARABIC LETTER SEEN
    'u0633': ('uFEB1', 'uFEB3', 'uFEB4', 'uFEB2'),
    # ARABIC LETTER SHEEN
    'u0634': ('uFEB5', 'uFEB7', 'uFEB8', 'uFEB6'),
    # ARABIC LETTER SAD
    'u0635': ('uFEB9', 'uFEBB', 'uFEBC', 'uFEBA'),
    # ARABIC LETTER DAD
    'u0636': ('uFEBD', 'uFEBF', 'uFEC0', 'uFEBE'),
    # ARABIC LETTER TAH
    'u0637': ('uFEC1', 'uFEC3', 'uFEC4', 'uFEC2'),
    # ARABIC LETTER ZAH
    'u0638': ('uFEC5', 'uFEC7', 'uFEC8', 'uFEC6'),
    # ARABIC LETTER AIN
    'u0639': ('uFEC9', 'uFECB', 'uFECC', 'uFECA'),
    # ARABIC LETTER GHAIN
    'u063A': ('uFECD', 'uFECF', 'uFED0', 'uFECE'),
    # ARABIC LETTER FEH
    'u0641': ('uFED1', 'uFED3', 'uFED4', 'uFED2'),
    # ARABIC LETTER QAF
    'u0642': ('uFED5', 'uFED7', 'uFED8', 'uFED6'),
    # ARABIC LETTER KAF
    'u0643': ('uFED9', 'uFEDB', 'uFEDC', 'uFEDA'),
    # ARABIC LETTER LAM
    'u0644': ('uFEDD', 'uFEDF', 'uFEE0', 'uFEDE'),
    # ARABIC LETTER MEEM
    'u0645': ('uFEE1', 'uFEE3', 'uFEE4', 'uFEE2'),
    # ARABIC LETTER NOON
    'u0646': ('uFEE5', 'uFEE7', 'uFEE8', 'uFEE6'),
    # ARABIC LETTER HEH
    'u0647': ('uFEE9', 'uFEEB', 'uFEEC', 'uFEEA'),
    # ARABIC LETTER WAW
    'u0648': ('uFEED', '', '', 'uFEEE'),
    # ARABIC LETTER (UIGHUR KAZAKH KIRGHIZ)? ALEF MAKSURA
    'u0649': ('uFEEF', 'uFBE8', 'uFBE9', 'uFEF0'),
    # ARABIC LETTER YEH
    'u064A': ('uFEF1', 'uFEF3', 'uFEF4', 'uFEF2'),
    # ARABIC LETTER ALEF WASLA
    'u0671': ('uFB50', '', '', 'uFB51'),
    # ARABIC LETTER U WITH HAMZA ABOVE
    'u0677': ('uFBDD', '', '', ''),
    # ARABIC LETTER TTEH
    'u0679': ('uFB66', 'uFB68', 'uFB69', 'uFB67'),
    # ARABIC LETTER TTEHEH
    'u067A': ('uFB5E', 'uFB60', 'uFB61', 'uFB5F'),
    # ARABIC LETTER BEEH
    'u067B': ('uFB52', 'uFB54', 'uFB55', 'uFB53'),
    # ARABIC LETTER PEH
    'u067E': ('uFB56', 'uFB58', 'uFB59', 'uFB57'),
    # ARABIC LETTER TEHEH
    'u067F': ('uFB62', 'uFB64', 'uFB65', 'uFB63'),
    # ARABIC LETTER BEHEH
    'u0680': ('uFB5A', 'uFB5C', 'uFB5D', 'uFB5B'),
    # ARABIC LETTER NYEH
    'u0683': ('uFB76', 'uFB78', 'uFB79', 'uFB77'),
    # ARABIC LETTER DYEH
    'u0684': ('uFB72', 'uFB74', 'uFB75', 'uFB73'),
    # ARABIC LETTER TCHEH
    'u0686': ('uFB7A', 'uFB7C', 'uFB7D', 'uFB7B'),
    # ARABIC LETTER TCHEHEH
    'u0687': ('uFB7E', 'uFB80', 'uFB81', 'uFB7F'),
    # ARABIC LETTER DDAL
    'u0688': ('uFB88', '', '', 'uFB89'),
    # ARABIC LETTER DAHAL
    'u068C': ('uFB84', '', '', 'uFB85'),
    # ARABIC LETTER DDAHAL
    'u068D': ('uFB82', '', '', 'uFB83'),
    # ARABIC LETTER DUL
    'u068E': ('uFB86', '', '', 'uFB87'),
    # ARABIC LETTER RREH
    'u0691': ('uFB8C', '', '', 'uFB8D'),
    # ARABIC LETTER JEH
    'u0698': ('uFB8A', '', '', 'uFB8B'),
    # ARABIC LETTER VEH
    'u06A4': ('uFB6A', 'uFB6C', 'uFB6D', 'uFB6B'),
    # ARABIC LETTER PEHEH
    'u06A6': ('uFB6E', 'uFB70', 'uFB71', 'uFB6F'),
    # ARABIC LETTER KEHEH
    'u06A9': ('uFB8E', 'uFB90', 'uFB91', 'uFB8F'),
    # ARABIC LETTER NG
    'u06AD': ('uFBD3', 'uFBD5', 'uFBD6', 'uFBD4'),
    # ARABIC LETTER GAF
    'u06AF': ('uFB92', 'uFB94', 'uFB95', 'uFB93'),
    # ARABIC LETTER NGOEH
    'u06B1': ('uFB9A', 'uFB9C', 'uFB9D', 'uFB9B'),
    # ARABIC LETTER GUEH
    'u06B3': ('uFB96', 'uFB98', 'uFB99', 'uFB97'),
    # ARABIC LETTER NOON GHUNNA
    'u06BA': ('uFB9E', '', '', 'uFB9F'),
    # ARABIC LETTER RNOON
    'u06BB': ('uFBA0', 'uFBA2', 'uFBA3', 'uFBA1'),
    # ARABIC LETTER HEH DOACHASHMEE
    'u06BE': ('uFBAA', 'uFBAC', 'uFBAD', 'uFBAB'),
    # ARABIC LETTER HEH WITH YEH ABOVE
    'u06C0': ('uFBA4', '', '', 'uFBA5'),
    # ARABIC LETTER HEH GOAL
    'u06C1': ('uFBA6', 'uFBA8', 'uFBA9', 'uFBA7'),
    # ARABIC LETTER KIRGHIZ OE
    'u06C5': ('uFBE0', '', '', 'uFBE1'),
    # ARABIC LETTER OE
    'u06C6': ('uFBD9', '', '', 'uFBDA'),
    # ARABIC LETTER U
    'u06C7': ('uFBD7', '', '', 'uFBD8'),
    # ARABIC LETTER YU
    'u06C8': ('uFBDB', '', '', 'uFBDC'),
    # ARABIC LETTER KIRGHIZ YU
    'u06C9': ('uFBE2', '', '', 'uFBE3'),
    # ARABIC LETTER VE
    'u06CB': ('uFBDE', '', '', 'uFBDF'),
    # ARABIC LETTER FARSI YEH
    'u06CC': ('uFBFC', 'uFBFE', 'uFBFF', 'uFBFD'),
    # ARABIC LETTER E
    'u06D0': ('uFBE4', 'uFBE6', 'uFBE7', 'uFBE5'),
    # ARABIC LETTER YEH BARREE
    'u06D2': ('uFBAE', '', '', 'uFBAF'),
    # ARABIC LETTER YEH BARREE WITH HAMZA ABOVE
    'u06D3': ('uFBB0', '', '', 'uFBB1'),
}

PRESENTATION_TO_BASE = dict()
for base in BASE_TO_PRESENTATION:
    for presentation in BASE_TO_PRESENTATION[base]:
        if presentation == '':
            continue
        PRESENTATION_TO_BASE[presentation.lower()] = base.lower()


with open(input_file, 'r') as fh:
    for line in fh:
        utt, uttid = line.strip().split("(")
        uttid = uttid.strip(")")
        char_array = utt.split()
        new_char_array = []
        nop = 0
        for char in char_array:
            # FIRST!!! convert from presentation form to base form
            if char in PRESENTATION_TO_BASE:
                char = PRESENTATION_TO_BASE[char]
            # Done with base-form-conversion



            # Normalize different Ligatures of LAM_ALIF to simple LAM ALIF
            if char == "ufefb" or char == "ufef7" or char == "ufef9" or char == "ufef5":
                new_char_array.append("u0644")
                new_char_array.append("u0627")
            elif char == "u0621":
                # stray hamzas -> null
                nop += 1
            elif char == "u0623" or char == "u0625" or char == "u0622":
                # alef hamza combo -> alef
                new_char_array.append("u0627")
            elif char == "u0624":
                # waw hamza -> waw
                new_char_array.append("u0648")
            elif char == "u0626" or char == "u0649":
                # yaa substitutions all -> yaa
                new_char_array.append("u064a")
            elif char == "u0629":
                # taa marbuta -> haa
                new_char_array.append("u0647")
            elif char == "u064b" or char == "u064c" or char == "u064d" or char == "u064e" or char == "u064f" or char == "u0650" or char == "u0651" or char == "u0652" or char == "u0653" or char == "u0654" or char == "u0655":
                # vowels and hamza -> null
                nop += 1
            elif char == "u0640":
                # remove tatweel
                nop += 1
            elif char == "u0660":
                # normalize ASCII and Arabic numerals to a single form
                # change Arabic number to ASCII number
                new_char_array.append("u0030")
            elif char == "u0661":
                new_char_array.append("u0031")
            elif char == "u0662":
                new_char_array.append("u0032")
            elif char == "u0663":
                new_char_array.append("u0033")
            elif char == "u0664":
                new_char_array.append("u0034")
            elif char == "u0665":
                new_char_array.append("u0035")
            elif char == "u0666":
                new_char_array.append("u0036")
            elif char == "u0667":
                new_char_array.append("u0037")
            elif char == "u0668":
                new_char_array.append("u0038")
            elif char == "u0669":
                new_char_array.append("u0039")
            elif char == "u25cf" or char == "u2022" or char == "u2219":
                new_char_array.append("u002e")
            elif char == "u060c":
                # Change Arabic comma to Reular Comma
                new_char_array.append("u002c")
            else:
                # Otherwise just apapend char w/o modification
                new_char_array.append(char)

        # Remove spaces at beginning or end of line
        while len(new_char_array) > 0 and new_char_array[0] == 'u0020':
            new_char_array = new_char_array[1:]
        while len(new_char_array) > 0 and new_char_array[-1] == 'u0020':
            new_char_array = new_char_array[:-1]

        # Finally, print out line
        output_lines.append(("%s (%s)\n" % (" ".join(new_char_array), uttid)))

with open(output_file, 'w') as fh:
    for line in output_lines:
        fh.write(line)
