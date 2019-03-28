import sys
from textutils import *

file1path = sys.argv[1]
file2path = sys.argv[2]
ids = []
refs = []
hyps1 = []
hyps2 = []
with open(file1path,'r') as file1:
    for line in file1.readlines():
        if "id:" in line:
            ids.append(line.strip('\n'))
        if "REF:" in line:
            tokens = line.split(' ')[1:]
            convertedline = ""
            for token in tokens:
                if '*' in token:
                    continue
                if ' ' in token or token == '':
                    continue
                letters = token.split('_')
                for letter in letters:
                    convertedline += uxxxx_to_utf8(letter)
                #convertedline += " "
            refs.append(convertedline)
        if "HYP:" in line:
            tokens = line.split(' ')[1:]
            convertedline = ""
            for token in tokens:
                if '*' in token:
                    continue
                letters = token.split('_')
                if ' ' in token or token == '':
                    continue
                for letter in letters:
                    convertedline += uxxxx_to_utf8(letter)
                #convertedline += " "
            hyps1.append(convertedline)
with open(file2path,'r') as file2:
    for line in file2.readlines():
        if "HYP:" in line:
            tokens = line.split(' ')[1:]
            convertedline = ""
            for token in tokens:
                if '*' in token:
                    continue
                letters = token.split('_')
                if ' ' in token or token == '':
                    continue
                for letter in letters:
                    convertedline += uxxxx_to_utf8(letter)
                #convertedline += " "
            hyps2.append(convertedline)

for i, imgid in enumerate(ids):
    print('\n' + imgid)
    print("REF: ", refs[i])
    print("HYP1: ", hyps1[i])
    print("HYP2: ", hyps2[i])
