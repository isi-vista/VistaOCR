import json
import logging
import os

import cv2
import lmdb
import numpy as np
import math
import torch

from torch.utils.data import Dataset

logger = logging.getLogger('root')


class RimesDataset(Dataset):
    def __init__(self, data_dir, split, line_height, transforms):
        logger.info("Loading RIMES [%s] dataset..." % split)

        self.data_dir = data_dir
        self.split = split
        self.line_height = line_height
        
        self.preprocess = transforms

        # Read Dataset Description
        with open(os.path.join(data_dir, 'desc.json'), 'r') as fh:
            self.data_desc = json.load(fh)

        # Init alphabet
        self.init_alphabet()

        # Read LMDB image database
        self.lmdb_env = lmdb.Environment(os.path.join(data_dir, 'line-images.lmdb'), map_size=1e12, readonly=True)
        self.lmdb_txn = self.lmdb_env.begin(buffers=True)

        # Divide dataset into classes by width of images images for two purposes:
        #    (1) It is more efficient to group images of roughly the samze size in a minibatch,
        #        because that results in less padding, and thus less wasted computation, per minibatch
        #    (2) It is probably true (although we haven't shown this conclusively) that the system can
        #        learn faster by starting of on smaller images and working its way up to longer images
        
        # empiracally found group sizes
        self.size_group_limits = [350, 520, 650, 999999999]
        self.size_group_keys = self.size_group_limits
        self.size_groups = dict()
        self.size_groups_dict = dict()

        for cur_limit in self.size_group_limits:
            self.size_groups[cur_limit] = []
            self.size_groups_dict[cur_limit] = dict()


        for idx, entry in enumerate(self.data_desc[self.split]):
            # Now figure out which size-group it belongs in
            for cur_limit in self.size_group_limits:
                # normalize width based on line_height and then add to correspoding bucket
                if math.ceil(entry['width'] * float(self.line_height / entry['height'])) <= cur_limit:
                    self.size_groups[cur_limit].append(idx)
                    self.size_groups_dict[cur_limit][idx] = 1
                    break

        logger.info("Done.")

    def init_alphabet(self):
        unique_chars = set()
        for entry in self.data_desc['train'] + self.data_desc['validation'] + self.data_desc['test']:
            for char in entry['trans'].split():
                unique_chars.add(char)

        self.idx_to_char = ['<ctc-blank>', *sorted(unique_chars)]
        self.char_to_idx = dict(zip(self.idx_to_char, range(len(self.idx_to_char))))
        # print("len of alphabet:", len(self.char_to_idx))
        # print("alphabets:", self.char_to_idx)

    def __getitem__(self, index):
        entry = self.data_desc[self.split][index]

        max_width = 0
        for cur_limit in self.size_group_limits:
            if index in self.size_groups_dict[cur_limit]:
                max_width = cur_limit
                break

        img_bytes = np.asarray(self.lmdb_txn.get(entry['id'].encode('ascii')), dtype=np.uint8)
        line_image = cv2.imdecode(img_bytes, -1)
        line_image = self.preprocess(line_image)

        # Add padding up to max-width, so that we have consistent size for cudnn.benchmark to work with
        original_width = line_image.size(2)
        if max_width < self.size_group_limits[-1]:
            torch.backends.cudnn.benchmark = True
            line_image_padded = torch.zeros(line_image.size(0), line_image.size(1), max_width)
            line_image_padded[:, :, :line_image.size(2)] = line_image
        else:
            # For extra long lines we don't pad unless we have to, and we turn off cdnn.benhcmark
            torch.backends.cudnn.benchmark = False
            line_image_padded = line_image

        transcription = []
        for char in entry['trans'].split():
            transcription.append(self.char_to_idx[char])


        metadata = {
            'utt-id': entry['id'],
            'width': original_width
        }

        return line_image_padded, transcription, metadata

    def __len__(self):
        return len(self.data_desc[self.split])
