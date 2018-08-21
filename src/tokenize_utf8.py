import unicodedata
import sys
import textutils


input_file = sys.argv[1]
output_file = sys.argv[2]

punc =  set(chr(i) for i in range(sys.maxunicode)
            if unicodedata.category(chr(i)).startswith('P'))

currency_symbols =  set(chr(i) for i in range(sys.maxunicode)
                        if unicodedata.category(chr(i)) == "Sc")

with open(input_file, 'r') as fh, open(output_file, 'w') as fh_out:
    for line in fh:
        line = line.strip()

        last_char = None
        for char in line:
            if char in punc or char in currency_symbols:
                if last_char != " ":
                    fh_out.write(" ")
                fh_out.write(char)
                fh_out.write(" ")
                last_char = " "
            else:
                if not (char == " " and last_char == " "):
                    fh_out.write(char)
                    last_char = char

        fh_out.write("\n")
