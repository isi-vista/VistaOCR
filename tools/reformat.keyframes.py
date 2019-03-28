#!/bin/env python

import sys, os
import argparse
import numpy as np
import re
import warnings

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Reformat the Text Detection & OCR outputs.')
    parser.add_argument('-i', '--input-text', dest='text', type=argparse.FileType('r'),
                        help="The input ocr text file")
    parser.add_argument('-m', '--input-msb', dest='msb',
                        help="The input msb file")
    args = parser.parse_args()

    SHOT_PAT = re.compile('(.*?)\_(\d+)$')

    shot_map = {}
    with open(args.msb, 'r') as fh:
        # maps from randomID to shotID
        lines = [line.rstrip() for line in fh]
        for line in lines:
            items = line.split()
            match = SHOT_PAT.match(items[1])
            if match:
                shot_map[items[0]] = match.group(1)

        lines = [line.rstrip() for line in args.text]
        for line in lines:
            items = line.split(',')

            match = SHOT_PAT.match(items[0])
            if match:
                id = match.group(1)
            else:
                continue

            try:
                videoID = shot_map[id]
                shotID = shot_map[id] + '_' + items[1]
            except:
                warnings.warn('Cannot find video ID: ' + id)
                continue

            xs = list(map(int, items[2:10:2]))
            ys = list(map(int, items[3:10:2]))
            text = items[10]

            if text == "":
                continue

            minx = np.min(xs)
            maxx = np.max(xs)
            miny = np.min(ys)
            maxy = np.max(ys)

            out = [videoID, shotID, str(minx), str(miny), str(maxx), str(maxy), text]

            print(','.join(out))

if __name__ == '__main__':
    main()
