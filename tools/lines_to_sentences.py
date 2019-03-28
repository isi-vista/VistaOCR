#!/bin/env python

import sys, os
import argparse
import numpy as np
import itertools
from operator import itemgetter


def is_neighbor(line1, line2):
    x1 = line1[2:10:2]
    x2 = line2[2:10:2]
    y1 = line1[3:10:2]
    y2 = line2[3:10:2]

    t1 = np.min(y1)
    b1 = np.max(y1)
    t2 = np.min(y2)
    b2 = np.max(y2)

    h1 = b1 - t1
    h2 = b2 - t2

    if np.abs(t1-b2) > np.minimum(h1, h2) and np.abs(t2-b1) > np.minimum(h1, h2):
        return False

    l1 = np.min(x1)
    r1 = np.max(x1)
    l2 = np.min(x2)
    r2 = np.max(x2)

    w1 = r1 - l1
    w2 = r2 - l2

    if (r1-l2)>0 and (r2-l1)>0:
        o = w1 + w2 - (np.minimum(r1, r2) - np.minimum(l1, l2))
        if float(o) / np.minimum(w1, w2) > 0.5:
            return True
        else:
            return False
    else:
        return False

def find_parent(line, mask):
    return mask[tuple(line)]

def set_parent(line, parent, mask):
    mask[tuple(line)] = tuple(parent)

def is_root(line, mask):
    return find_parent(line, mask) == -1 

def find_root(line, mask):
    root = line
    update_parent = False
    while not is_root(root, mask):
        root = find_parent(root, mask)
        update_parent = True
    if update_parent:
        set_parent(line, root, mask)
    return root

def join(line1, line2, mask):
    box1 = line1[2:10]
    box2 = line2[2:10]
    root1 = find_root(box1, mask)
    root2 = find_root(box2, mask)

    if tuple(root1) != tuple(root2):
        set_parent(root1, root2, mask)

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Merge text lines to sentence')
    parser.add_argument('-i', '--input-text', dest='text',
                        help="The input ocr text")
    parser.add_argument('-o', '--out-file', dest='out_file', type=argparse.FileType('w'),
                        help="The output text file")
    args = parser.parse_args()

    dummy = {}
    with open(args.text, 'r') as fh:
        lines = [line.rstrip() for line in fh]

        for line in lines:
            line = line.split(',')
            line[2:10] = list(map(int, line[2:10]))

            # line[1] is frame id, line[3] is x
            if line[1] in dummy:
                dummy[line[1]][line[3]] = line
            else:
                dummy[line[1]] = {line[3]: line}

    frames = {}
    for frame_id in dummy:
        for key, value in sorted(dummy[frame_id].items()):
            if frame_id in frames:
                frames[frame_id].append(value)
            else:
                frames[frame_id] = [value]

    for frame_id in frames:
        mask = {}
        for line in frames[frame_id]:
            mask[tuple(line[2:10])] = -1
            
        for id1 in range(len(frames[frame_id])-1):
            for id2 in range(id1+1, len(frames[frame_id])):
                if is_neighbor(frames[frame_id][id1], frames[frame_id][id2]):
                    join(frames[frame_id][id1], frames[frame_id][id2], mask)

        root_map = {}
        group = {}
        def get_index(root):
            if tuple(root) not in root_map:
                root_map[tuple(root)] = len(root_map)+1
            return root_map[tuple(root)]

        for line in frames[frame_id]:
            root = find_root(line[2:10], mask)
            idx = get_index(root)
            group[tuple(line[2:10])] = idx
        
        text_map = {}
        for num, line in enumerate(frames[frame_id]):
            text_map[tuple(line[2:10])] = num

        line_group = {}
        for line in group:
            if group[line] in line_group:
                line_group[group[line]] += [frames[frame_id][text_map[line]]]
            else:
                line_group[group[line]] = [frames[frame_id][text_map[line]]]


        lines = []
        for id in line_group:
            sorted_line = sorted(line_group[id], key=itemgetter(3))

            minx = np.min([ x[2:10:2] for x in line_group[id] ])
            maxx = np.max([ x[2:10:2] for x in line_group[id] ])
            miny = np.min([ x[3:10:2] for x in line_group[id] ])
            maxy = np.max([ x[3:10:2] for x in line_group[id] ])

            text = ' '.join([x[11] for x in line_group[id]])

            lines.append([','.join(line_group[id][0][0:2]), minx, maxy, minx, miny, maxx, miny, maxx, maxy, text])

        frames[frame_id] = lines

    for key in frames:
        for line in frames[key]:
            text = ','.join(list(map(str, line)))
            args.out_file.write(text+'\n')

if __name__ == '__main__':
    main()
