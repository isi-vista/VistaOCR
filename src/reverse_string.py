import sys

if len(sys.argv) != 2:
    print("Usage: reverse_string.py input-file")
    sys.exit(1)

input_file = sys.argv[1]

with open(input_file, 'r') as fh:
    for line in fh:
        utt, uttid = line.strip().split("(")
        uttid = uttid.strip(")")

        char_array = utt.split()
        reversed_utt = ' '.join(list(reversed(char_array)))
        print("%s (%s)" % (reversed_utt, uttid))

