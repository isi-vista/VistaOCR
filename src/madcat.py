import json
import logging
import sys
import os
import math
import cv2
import lmdb
import numpy as np
import torch
import textutils
import arabic_reshaper

from torch.utils.data import Dataset

logger = logging.getLogger('root')


class MadcatDataset(Dataset):
    def __init__(self, data_dir, split, line_height, transforms, seq2seq=False):
        logger.info("Loading MADCAT [%s] dataset..." % split)

        self.seq2seq = seq2seq

        self.data_dir = data_dir
        self.split = split
        self.line_height = line_height

        self.preprocess = transforms

        self.mean = 0.100589
        self.stdv = 0.294157

        # Read Dataset Description
        #with open(os.path.join(data_dir, 'desc-presentationform.json'), 'r') as fh:
        with open(os.path.join(data_dir, 'desc-fair.json'), 'r') as fh:
            self.data_desc = json.load(fh)

        # Init alphabet
        self.init_alphabet()

        # Read LMDB image database
        self.lmdb_env = lmdb.Environment(os.path.join(data_dir, 'line-images.lmdb'), map_size=1e12, readonly=True)
        self.lmdb_txn = self.lmdb_env.begin(buffers=True)

        # Potentially read writer-id features
        if os.path.exists(os.path.join(data_dir, 'wid-o.lmdb')):
            self.wid_lmdb_env = lmdb.Environment(os.path.join(data_dir, 'wid-o.lmdb'), map_size=1e12, readonly=True)
            self.wid_lmdb_txn = self.wid_lmdb_env.begin(buffers=True)
        else:
            self.wid_lmdb_txn = None

        # Divide dataset into classes by width of images images for two purposes:
        #    (1) It is more efficient to group images of roughly the samze size in a minibatch,
        #        because that results in less padding, and thus less wasted computation, per minibatch
        #    (2) It is probably true (although we haven't shown this conclusively) that the system can
        #        learn faster by starting of on smaller images and working its way up to longer images
        #
        # We emprically measure that for MADCAT data, the breakdown in line-widths is roughly:
        #    Width Range      Cumualtive Percent of Data
        #       0-150              10%
        #     150-200              20%
        #     200-300              45%
        #     300-350              70%
        #     350-450              90%
        #     450-600              99%
        #     600+                 100%

        self.size_group_limits = [150, 200, 300, 350, 450, 600, 999999999]

        if self.split == "test":
            # For test set, want to run every image, so include extra-long lines
            self.size_group_keys = self.size_group_limits
        else:
            # For training/val sets throw away extra-long lines to reduce memory pressure
            self.size_group_keys = self.size_group_limits[0:-1]

        self.size_groups = dict()
        self.size_groups_dict = dict()

        for cur_limit in self.size_group_limits:
            self.size_groups[cur_limit] = []
            self.size_groups_dict[cur_limit] = dict()

        # First handle writer id
        self.writer_id_map = dict()
        # Want to define order instead of using random json order
        for mode in ['train', 'validation', 'test']:
            for entry in self.data_desc[mode]:
                if not entry['writer'] in self.writer_id_map:
                    self.writer_id_map[entry['writer']] = len(self.writer_id_map)

        # Now handle size-groups
        for idx, entry in enumerate(self.data_desc[self.split]):
            for cur_limit in self.size_group_limits:
                if entry['width'] < cur_limit:
                    self.size_groups[cur_limit].append(idx)
                    self.size_groups_dict[cur_limit][idx] = 1
                    break

        logger.info("Done.")


    def init_alphabet(self):
        unique_chars = set()
        for entry in self.data_desc['train']:
            for char in entry['trans'].split():
                unique_chars.add(char)


        if self.seq2seq:
            self.idx_to_char = ['<pad>', '<s>', '</s>', '<unk>', *sorted(unique_chars)]
        else:
            self.idx_to_char = ['<ctc-blank>', *sorted(unique_chars)]

        self.char_to_idx = dict(zip(self.idx_to_char, range(len(self.idx_to_char))))


    def __getitem__(self, index):
        return self.get_item(index, flip=True)

    def get_item(self, index, flip=True):
        entry = self.data_desc[self.split][index]
        max_width = 0
        for cur_limit in self.size_group_limits:
            if index in self.size_groups_dict[cur_limit]:
                max_width = int(math.ceil(cur_limit * self.line_height/30))
                break

        img_bytes = np.asarray(self.lmdb_txn.get(entry['id'].encode('ascii')), dtype=np.uint8)
        line_image = cv2.imdecode(img_bytes, -1)
        # Flip the line image, because arabic is rtl, but CTC assumes a LTR decoding
        if flip:
            line_image = cv2.flip(line_image, flipCode=1)

        line_image = self.preprocess(line_image)


        # Add padding up to max-width, so that we have consistent size for cudnn.benchmark to work with
        original_width = line_image.size(2)
        if max_width < self.size_group_limits[-1]:
            torch.backends.cudnn.benchmark = True
            line_image_padded = torch.zeros(line_image.size(0), line_image.size(1), max_width)
            line_image_padded[:, :, :line_image.size(2)] = line_image
        else:
            # For extra long lines we don't pad unless we have to, and we turn off cdnn.benhcmark
            #torch.backends.cudnn.benchmark = False
            line_image_padded = line_image

        transcription = []

        if self.seq2seq:
            transcription.append(self.char_to_idx['<s>'])
        for char in entry['trans'].split():
            transcription.append(self.char_to_idx[char])

        if self.seq2seq:
            transcription.append(self.char_to_idx['</s>'])

        metadata = {
            'writer-id': self.writer_id_map[entry['writer']],
            'utt-id': entry['id'],
            'width': original_width
        }

        if not self.wid_lmdb_txn is None:
            wid_fv = np.frombuffer(self.wid_lmdb_txn.get(entry['id'].encode('ascii')), dtype=np.float32)
            metadata['wid-feat'] = torch.from_numpy(wid_fv)

        return line_image_padded, transcription, metadata

    def __len__(self):
        size = 0
        for i in range(len(self.size_group_limits) - 1): # = [150, 200, 300, 350, 450, 600, 999999999]
            size += len(self.size_groups[self.size_group_limits[i]])
        return size
        #return len(self.data_desc[self.split])
