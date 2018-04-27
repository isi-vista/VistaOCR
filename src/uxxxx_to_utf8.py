import sys

from english import EnglishAlphabet
from textutils import form_tokenized_words

input_file = sys.argv[1]
output_file = sys.argv[2]




output_lines = []
with open(input_file, "r") as fh:
    for line in fh:
        utt, uttid = line.strip().split("(")
        uttid = uttid.strip(")")

        idx = 0
        cur_token_str = ''
        # String either looks like:
        #   u0032 u0035 u0020 u0031 u0030
        # Or
        #   u0032_u0035 u0020 u0031_u0030
        while idx < len(utt):
            # Sanity check
            if utt[idx] == ' ':
                cur_token_str += ' '
                idx += 1
            elif utt[idx] == '_':
                idx += 1
            if idx >= len(utt):
                break
            if not utt[idx] == 'u':
                raise Exception("Incorrectly formatted line; utt[%d] = %s;  line = %s" % (idx, utt[idx], line))

            # First get the 'xxxx' part out of the current 'uxxxx' char
            cur_char = utt[(idx+1):(idx+5)]

            if cur_char == "0020":
                cur_token_str += "^"
            else:
                cur_token_str += chr(int(cur_char, 16))
        
            # Advance idx to ppoint to next 'uxxxx' char
            idx += 5

        output_lines.append(("%s (%s)\n" % (cur_token_str, uttid)))

with open(output_file, "w") as fh:
    for line in output_lines:
        fh.write(line)
