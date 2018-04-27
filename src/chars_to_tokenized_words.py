import sys

from textutils import form_tokenized_words

input_file = sys.argv[1]
output_file = sys.argv[2]

output_lines = []
with open(input_file, "r") as fh:
    for line in fh:
        utt, uttid = line.strip().split("(")
        uttid = uttid.strip(")")
        tokenized_utt = ' '.join(form_tokenized_words(utt.split()))
        output_lines.append(("%s (%s)\n" % (tokenized_utt, uttid)))

with open(output_file, "w") as fh:
    for line in output_lines:
        fh.write(line)
