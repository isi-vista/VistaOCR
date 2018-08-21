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

from ocr_dataset import OcrDataset

class OcrDatasetUnion(Dataset):
    def __init__(self, data_dir_list, split, transforms, alphabet=None):
        logger.info("Loading OCR Dataset Union: [%s] split from [%s]." % (split, data_dir_list))

        self.datasets = []
        self.nentries = 0

        self.lmdb_cache = dict()
        for data_dir in data_dir_list:
            if data_dir in self.lmdb_cache:
                dataset = OcrDataset(data_dir, split, transforms, alphabet, preloaded_lmdb=self.lmdb_cache[data_dir])
            else:
                dataset = OcrDataset(data_dir, split, transforms, alphabet)
                self.lmdb_cache[data_dir] = dataset.lmdb_env
            self.datasets.append(dataset)
            self.nentries += len(dataset)

        # Because different datasets might have different alphabets, we need to merge them and unify
        self.merge_alphabets()

        # Merge size group stuff (ugh)
        self.size_group_keys = self.datasets[0].size_group_keys
        self.size_groups = dict()
        for cur_limit in self.size_group_keys:
            self.size_groups[cur_limit] = []


        accumulatd_max_idx = 0
        for ds in self.datasets:
            # For now we only merge if szme set of size groups  (need to change this requirement!)
            assert ds.size_group_keys == self.size_group_keys
            for cur_limit in self.size_group_keys:
                self.size_groups[cur_limit].extend([accumulatd_max_idx + idx for idx in ds.size_groups[cur_limit]])
            accumulatd_max_idx += ds.max_index

            


    def merge_alphabets(self):
        alphabet_list = [ds.alphabet for ds in self.datasets]
        unique_chars = set()
        for alphabet in alphabet_list:
            # First entry is always <ctc-blank>, so let's just grab all the entries after that
            unique_chars.update(alphabet.char_array[1:])

        # Now create a new alphabet w/ merged characters
        self.alphabet = Alphabet(['<ctc-blank>', *sorted(unique_chars)])

        # And propogate it to each of our datasets
        for ds in self.datasets:
            ds.alphabet = self.alphabet


    def __getitem__(self, index):
        accumulatd_max_idx = 0
        for dataset in self.datasets:
            if index <= accumulatd_max_idx + dataset.max_index:
                return dataset[index - accumulatd_max_idx]
            accumulatd_max_idx += dataset.max_index

        print("index = %d" % index)
        print("total num entries = %d" % self.nentries)
        print("Size of each dataset = ")
        for ds in self.datasets:
            print("Size = %d" % len(ds))
        assert False, "Should never get here"

    def __len__(self):
        return self.nentries

