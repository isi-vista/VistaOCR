import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

output_lines = []
with open(input_file, "r") as fh:
    for line in fh:
        utt, uttid = line.strip().split("(")
        uttid = uttid.strip(")")

        char_array = []

        prev_was_space = False
        for char in utt.split():
            if char != 'u0020':
                char_array.append(char)
                prev_was_space = False

            if char == 'u0020' and not prev_was_space:
                char_array.append(char)
                prev_was_space = True
            

        # Also don't want to start or end on space
        if len(char_array) > 0:
            if char_array[0] == 'u0020':
                char_array = char_array[1:]

        if len(char_array) > 0:
            if char_array[-1] == 'u0020':
                char_array = char_array[:-1]

        new_utt = ' '.join(char_array)
        output_lines.append(("%s (%s)\n" % (new_utt, uttid)))

with open(output_file, "w") as fh:
    for line in output_lines:
        fh.write(line)
