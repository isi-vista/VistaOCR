import sys
import textutils


input_file = sys.argv[1]

with open(input_file, 'r') as fh:

    for line in fh:
        lparen_location = line.rfind('(')
        rparen_location = line.rfind(')')

        utt = line[ :lparen_location]
        utt_utf8 = ''
        for word in utt.split(" "):
            if word == "u0020":
                utt_utf8 += " "
            elif word == "u0009":
                utt_utf8 += " "
            else:
                word_utf8 = ''
                for char in word.split("_"):
                    char_utf8 = textutils.uxxxx_to_utf8(char)
                    word_utf8 += char_utf8

                utt_utf8 += word_utf8

        uttid = line[ lparen_location+1 : rparen_location ]
        uttid = uttid[ :uttid.rfind('_') ]

        print("%s (%s)" % (utt_utf8, uttid))

