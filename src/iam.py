import json
import logging
import os

import cv2
import lmdb
import numpy as np
import torch

logger = logging.getLogger('root')

from torch.utils.data import Dataset

class IAMDataset(Dataset):
    def __init__(self, data_dir, split, alphabet, line_height, transforms):
        logger.info("Loading IAM [%s] dataset..." % split)

        self.data_dir = data_dir
        self.split = split
        self.alphabet = alphabet

        self.preprocess = transforms

        # Read Dataset Description
        with open(os.path.join(data_dir, 'desc.json'), 'r') as fh:
            self.data_desc = json.load(fh)

        # Read LMDB image database
        self.lmdb_env = lmdb.Environment(os.path.join(data_dir, 'line-images.lmdb'), map_size=1e12, readonly=True)
        self.lmdb_txn = self.lmdb_env.begin(buffers=True)

        # Divide dataset into classes by width of images images for two purposes:
        #    (1) It is more efficient to group images of roughly the samze size in a minibatch,
        #        because that results in less padding, and thus less wasted computation, per minibatch
        #    (2) It is probably true (although we haven't shown this conclusively) that the system can
        #        learn faster by starting of on smaller images and working its way up to longer images
        #
        # We emprically measure that for IAM data, the breakdown in line-widths is roughly:
        #    Width Range      Cumualtive Percent of Data
        #       0-300              10%
        #     300-350              20%
        #     350-400              40%
        #     400-500              70%
        #     500-600              90%
        #     600+                 100%

        self.size_group_limits = [400, 600, 999999999]
        self.size_group_keys = self.size_group_limits
        self.size_groups = dict()
        self.size_groups_dict = dict()

        for cur_limit in self.size_group_limits:
            self.size_groups[cur_limit] = []
            self.size_groups_dict[cur_limit] = dict()

        self.writer_id_map = dict()

        for idx, entry in enumerate(self.data_desc[self.split]):
            # First handle writer id
            if not entry['writer'] in self.writer_id_map:
                self.writer_id_map[entry['writer']] = len(self.writer_id_map)

            # Now figure out which size-group it belongs in
            for cur_limit in self.size_group_limits:
                if entry['width'] < cur_limit:
                    self.size_groups[cur_limit].append(idx)
                    self.size_groups_dict[cur_limit][idx] = 1
                    break

        logger.info("Done.")

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
            line_image_padded = line_image[:, :, :line_image.size(2)]
        else:
            # For extra long lines we don't pad unless we have to, and we turn off cdnn.benhcmark
            torch.backends.cudnn.benchmark = False
            line_image_padded = line_image

        transcription = []
        for char in entry['trans'].split(" "):
            transcription.append(self.alphabet.alphabet_to_idx[char.lower()])

        metadata = {
            'writer-id': self.writer_id_map[entry['writer']],
            'utt-id': entry['id'],
            'width': original_width
        }

        return line_image_padded, transcription, metadata

    def __len__(self):
        return len(self.data_desc[self.split])
