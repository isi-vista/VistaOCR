import logging
import math
import time
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from textutils import uxxxx_to_utf8

from torch.nn.utils.rnn import pack_padded_sequence as rnn_pack
from torch.nn.utils.rnn import pad_packed_sequence as rnn_unpack

logger = logging.getLogger('root')


class CnnScriptIdModelMean(nn.Module):
    def get_hyper_params(self):
        return self.hyper_params

    @classmethod
    def FromSavedWeights(cls, weight_file, verbose=True, gpu=None):
        weights = torch.load(weight_file)

        if verbose:
            logger.info("Loading model from: %s" % weight_file)
            logger.info("\tFrom iteration: %d" % weights['iteration'])
            #logger.info("\tWithout LM: Val CER: %.2f\tWER: %.2f" % (100 * weights['val_cer'], 100 * weights['val_wer']))
            logger.info("\tModel Hyperparams = %s" % str(weights['model_hyper_params']))
            logger.info("")

        hp = weights['model_hyper_params']

        # Need to over-ride GPU parameter
        if not gpu is None:
            hp['gpu'] = gpu
        hp['verbose'] = verbose

        # Create model with right hyperparametrs
        model = cls(**hp)

        # Now load weights from previously trained model
        model.load_state_dict(weights['state_dict'], strict=False)

        return model

    def __init__(self, *args, **kwargs):
        super(CnnScriptIdModelMean, self).__init__()

        if len(args) > 0:
            raise Exception("Only keyword arguments allowed in CnnScriptIdModel")

        self.hyper_params = kwargs.copy()

        self.input_line_height = kwargs['input_line_height']

        self.num_in_channels = kwargs.get('num_in_channels', 1)

        self.gpu = kwargs.get('gpu', True)
        self.multigpu = kwargs.get('multigpu', True)
        self.verbose = kwargs.get('verbose', True)

        self.cnn = nn.Sequential(
            *self.ConvBNReLU(3, 64),
            *self.ConvBNReLU(64, 64),
            nn.FractionalMaxPool2d(2, output_ratio=(0.5, 0.7)),
            *self.ConvBNReLU(64, 128),
            *self.ConvBNReLU(128, 128),
            nn.FractionalMaxPool2d(2, output_ratio=(0.5, 0.7)),
            *self.ConvBNReLU(128, 256),
            *self.ConvBNReLU(256, 256),
            *self.ConvBNReLU(256, 256)
        )


        # For now, only predicting Is input Cyrillic? if not then it must Latin (aka enlgish)
        cnn_output_h = int(self.input_line_height / 4)
        self.linear = nn.Linear(256 * cnn_output_h, 2)

        # Finally, let's initialize parameters
        for param in self.parameters():
            torch.nn.init.uniform(param, -0.08, 0.08)

        if torch.cuda.is_available() and self.gpu:
            self.cnn = self.cnn.cuda()
            self.linear = self.linear.cuda()
            if self.multigpu:
                self.cnn = torch.nn.DataParallel(self.cnn)
        else:
            logger.info("Warning: Runnig model on CPU")

    # Consider: nn.Dropout2d
    def ConvBNReLU(self, nInputMaps, nOutputMaps):
        return [nn.Conv2d(nInputMaps, nOutputMaps, kernel_size=3, padding=1),
                nn.BatchNorm2d(nOutputMaps),
                nn.ReLU(inplace=True)]

    def forward(self, x):
        cnn_output = self.cnn(x)
        b, c, h, w = cnn_output.size()
        #cnn_output = cnn_output.permute(3, 0, 1, 2).contiguous()
        # Now collapse output to a single timestep using avg pooling
        pooled_feat = torch.mean(cnn_output, dim=-1)
        # Now feed into linear classification layer
        classification_output = self.linear(pooled_feat.view(b, -1))

        return classification_output


