import sys
from textutils import *

inpath = sys.argv[1]
#outpath = sys.argv[2]
uni_punctuation = []

punctuation = ['.',',','/','!','`','~','>','<','[',']','{','}','#','@','$','5','^','&','*','(',')','-','_','+','=','\\','\'','\"',':','?']
for p in punctuation:
    uni_punctuation.append(utf8_to_uxxxx(p))
with open(inpath,'r') as infile:
    for line in infile.readlines():
        for up in uni_punctuation:
            line = line.replace(' ' + up, '')
            line = line.replace(up + ' ', '')
            line = line.replace(up, '')
        sys.stdout.write(line)
        
