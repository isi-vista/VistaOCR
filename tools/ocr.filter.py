#!/bin/env python

import sys, os
import re
import codecs
import argparse
import numpy as np


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Filtering OCR outputs')
    parser.add_argument('-i', '--input-text', dest='text', type=argparse.FileType('r'),
                        help="The input ocr text")
    args = parser.parse_args()

    volcabulary = set()
    for line in args.text:
        items = line.rstrip().upper().split(',')
        words = items[6].split()
        count = 0

        if len(words) <= 3:
            continue

        for w in words:
            if len(w) <= 1:
                count += 1
        if float(count) / float(len(words)) > 0.3:
            continue
        else:
            print(line.rstrip()) 

if __name__ == '__main__':
    main()
