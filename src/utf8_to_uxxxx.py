import sys
import textutils


input_file = sys.argv[1]

with open(input_file, 'r') as fh:

    for line in fh:
        lparen_location = line.rfind('(')
        rparen_location = line.rfind(')')

        utt = line[ :lparen_location]
        utt_utf8 = textutils.utf8_to_uxxxx(utt)

        uttid = line[ lparen_location+1 : rparen_location ]

        print("%s (%s)" % (utt_utf8, uttid))

