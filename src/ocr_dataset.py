import json
import logging
import os

import cv2
import lmdb
import numpy as np
import torch

logger = logging.getLogger('root')

from torch.utils.data import Dataset
from alphabet import Alphabet


class OcrDataset(Dataset):
    def __init__(self, data_dir, split, transforms, alphabet=None, preloaded_lmdb=None, max_allowed_width=1200):
        logger.info("Loading OCR Dataset: [%s] split from [%s]." % (split, data_dir))

        self.max_allowed_width = max_allowed_width

        self.data_dir = data_dir
        self.split = split
        self.preprocess = transforms

        # Read Dataset Description
        with open(os.path.join(data_dir, 'desc.json'), 'r') as fh:
            self.data_desc = json.load(fh)


        # Init alphabet
        if alphabet is not None:
            logger.info("Explicitly providing alphabet via initializatin parameter, as opposed to inferring from data")
            self.alphabet = alphabet
        else:
            self.init_alphabet()

        # Read LMDB image database
        if preloaded_lmdb:
            self.lmdb_env = preloaded_lmdb
        else:
            self.lmdb_env = lmdb.Environment(os.path.join(data_dir, 'line-images.lmdb'), map_size=1e12, readonly=True, lock=False)

        self.lmdb_txn = self.lmdb_env.begin(buffers=True)

        # Divide dataset into classes by width of images images for two purposes:
        #    (1) It is more efficient to group images of roughly the samze size in a minibatch,
        #        because that results in less padding, and thus less wasted computation, per minibatch
        #    (2) It is probably true (although we haven't shown this conclusively) that the system can
        #        learn faster by starting of on smaller images and working its way up to longer images
        #

        # TODO:  write determine_width_cutoffs() to determine best answer automatically
        #        For now just hard-code some heuristic approach
        #if len(self.data_desc['train']) < 10000:
        #    self.size_group_limits = [400, 600, np.inf]
        #else:
        #    self.size_group_limits = [150, 200, 300, 350, 450, 600, np.inf]

        self.size_group_limits = [150, 200, 300, 350, 450, 600, np.inf]


        self.size_group_keys = self.size_group_limits
        self.size_groups = dict()
        self.size_groups_dict = dict()

        for cur_limit in self.size_group_limits:
            self.size_groups[cur_limit] = []
            self.size_groups_dict[cur_limit] = dict()

        self.writer_id_map = dict()

        for idx, entry in enumerate(self.data_desc[self.split]):
            # First handle writer id
            if 'writer' in entry:
                if not entry['writer'] in self.writer_id_map:
                    self.writer_id_map[entry['writer']] = len(self.writer_id_map)

            # Now figure out which size-group it belongs in
            for cur_limit in self.size_group_limits:
                if ('height' in entry) and ('width' in entry):
                    width_orig, height_orig = entry['width'], entry['height']
                    normalized_height = 30
                    normalized_width = width_orig * (normalized_height / height_orig)
                elif 'width' in entry:
                    normalized_width = entry['width']
                else:
                    raise Exception("Json entry must list width & height of image.")

                if normalized_width < cur_limit and normalized_width < self.max_allowed_width:
                    self.size_groups[cur_limit].append(idx)
                    self.size_groups_dict[cur_limit][idx] = 1
                    break


        # Now get final size (might have dropped large entries!)
        self.nentries = 0
        self.max_index = 0
        for cur_limit in self.size_group_limits:
            self.nentries += len(self.size_groups[cur_limit])

            if len(self.size_groups[cur_limit]) > 0:
                cur_max = max(self.size_groups[cur_limit])
                if cur_max > self.max_index:
                    self.max_index = cur_max

        logger.info("Done.")

    def init_alphabet(self):
        # Read entire train/val/test data to deterimine set of unique characters we should have in alphabet
        unique_chars = set()

        for split in ['train', 'validation', 'test']:
            for entry in self.data_desc[split]:
                for char in entry['trans'].split():
                    unique_chars.add(char)


        # Now add CTC blank as first letter in alphabet. Also sort alphabet lexigraphically for convinience
        self.alphabet = Alphabet(['<ctc-blank>', *sorted(unique_chars)])

    def determine_width_cutoffs(self):

        # The purpsoe of this function is to break the data into groups of similar widths
        # for example:  all images of width between 0 and 100; all images of widths between 101 and 200; etc
        #
        # We do this for performance reasons as mentioned above; we set the group cutoffs based on the dataset

        # First let's cycle through data and get count of all widths:
        max_normalized_width = 800
        normalized_height = 30
        normalized_width_counts = np.zeros((max_normalized_width))

        remove_count = 0
        for idx, entry in enumerate(self.data_desc[self.split]):
            w, h = entry['width'], entry['height']
            normalized_width = (normalized_height/h) * w

            if normalized_width > max_normalized_width:
                remove_count += 1
            else:
                normalized_width_counts[normalized_width-1] += 1


        logger.info("Removed %d images due to max width cutoff" % remove_count)

        # Now use data to determine cutoff points
        normalized_width_cumsum = np.cumsum(normalized_width_counts)

        return None


    def __getitem__(self, index):
        entry = self.data_desc[self.split][index]
        max_width = 0
        for cur_limit in self.size_group_limits:
            if index in self.size_groups_dict[cur_limit]:
                max_width = cur_limit
                break

        img_bytes = np.asarray(self.lmdb_txn.get(entry['id'].encode('ascii')), dtype=np.uint8)
        line_image = cv2.imdecode(img_bytes, -1)

#        # HACK for now to flip right-to-left
#        if 'farsi' in self.data_dir or 'arabic' in self.data_dir:
#            line_image = cv2.flip(line_image, flipCode=1)

        # Do a check for RGBA images; if found get rid of alpha channel
        if len(line_image.shape) == 3 and line_image.shape[2] == 4:
            line_image = cv2.cvtColor(line_image, cv2.COLOR_BGRA2BGR)

        if line_image.shape[0] == 0 or line_image.shape[1] == 0:
            print("ERROR, line image is 0 area; id = %s; idx = %d" % (entry['id'], index) )

        line_image = self.preprocess(line_image)

        # Sanity check: make sure width@30px lh is long enough not to crash our model; we pad to at least 15px wide
        # Need to do this and change the "real" image size so that pack_padded doens't complain
        if line_image.size(2) < 15:
            line_image_ = torch.ones(line_image.size(0), line_image.size(1), 15)
            line_image_[:,:,:line_image.size(2)] = line_image
            line_image = line_image_

        # Add padding up to max-width, so that we have consistent size for cudnn.benchmark to work with
        original_width = line_image.size(2)
        if max_width < self.size_group_limits[-1]:
            torch.backends.cudnn.benchmark = True
            line_image_padded = torch.zeros(line_image.size(0), line_image.size(1), max_width)
            line_image_padded = line_image[:, :, :line_image.size(2)]
        else:
            # For extra long lines we don't pad unless we have to, and we turn off cdnn.benhcmark
            # Note: Problem--if this is on seperate process via MultiProcess library, then this doesn't actually affect anything!
            torch.backends.cudnn.benchmark = False
            line_image_padded = line_image

        transcription = []
        for char in entry['trans'].split():
            transcription.append(self.alphabet.char_to_idx[char])

        metadata = {
            'utt-id': entry['id'],
            'width': original_width
        }

        if 'writer' in entry:
            metadata['writer-id'] = self.writer_id_map[entry['writer']]


        return line_image_padded, transcription, metadata

    def __len__(self):
        return self.nentries

