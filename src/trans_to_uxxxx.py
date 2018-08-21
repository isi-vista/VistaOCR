import sys

input_file = sys.argv[1]

def utf8_to_uxxxx(in_str):

    char_array = []
    out_str = ""
    for char in in_str:
        if char == " ":
            out_str += '_'.join(char_array) + " "
            char_array = []
        else:
            raw_hex = hex(ord(char))[2:].zfill(4).lower()
            char_array.append( "u%s" % raw_hex )

    if len(char_array) > 0:
        out_str += '_'.join(char_array)

    return out_str

with open(input_file, 'r') as fh:

    for line in fh:
        lparen_location = line.rfind('(')
        rparen_location = line.rfind(')')

        utt = line[ :lparen_location]

        utt_uxxxx = utf8_to_uxxxx(utt)

        uttid = line[ lparen_location+1 : rparen_location ]
        print("%s (%s)" % (utt_uxxxx, uttid))

