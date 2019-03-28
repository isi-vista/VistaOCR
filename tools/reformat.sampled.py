#!/bin/env python

import sys, os
import argparse
import numpy as np
import re


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Reformat the Text Detection & OCR outputs.')
    parser.add_argument('-i', '--input-text', dest='text', type=argparse.FileType('r'),
                        help="The input ocr text file")
    args = parser.parse_args()

    SHOT_PAT = re.compile('(.*?)\_(\d+)$') 

    shot_map = {}
    lines = [line.rstrip() for line in args.text]
    for line in lines:
        items = line.split(',')

        videoID = items[0]
        frameID = items[1]
        text    = items[10]
        shotID  = videoID + '_' + frameID
        out = [videoID, shotID, "-1", "-1", "-1", "-1", text]
        print(','.join(out))

if __name__ == '__main__':
    main()
