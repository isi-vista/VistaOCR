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


class CnnScriptIdLstmModel(nn.Module):
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
        super(CnnScriptIdLstmModel, self).__init__()

        if len(args) > 0:
            raise Exception("Only keyword arguments allowed in CnnScriptIdModel")

        self.hyper_params = kwargs.copy()

        self.input_line_height = kwargs['input_line_height']

        self.lstm_input_dim = kwargs['lstm_input_dim']
        self.num_lstm_layers = kwargs['num_lstm_layers']
        self.num_lstm_hidden_units = kwargs['num_lstm_hidden_units']
        self.p_lstm_dropout = kwargs['p_lstm_dropout']
        self.num_in_channels = kwargs.get('num_in_channels', 1)

        self.gpu = kwargs.get('gpu', True)
        self.multigpu = kwargs.get('multigpu', True)
        self.verbose = kwargs.get('verbose', True)
        #ADDED
        self.LSTMList = None
        self.lattice_decoder = None
        #/ADDED#
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
        #ADDED
        # We need to calculate cnn output size to construct the bridge layer
        fake_input_width = 20
        cnn_out_h, _ = self.cnn_input_size_to_output_size((self.input_line_height, fake_input_width))
        cnn_out_c = self.cnn_output_num_channels()
        cnn_feat_size = cnn_out_c * cnn_out_h

        self.bridge_layer = nn.Sequential(
            nn.Linear(cnn_feat_size, self.lstm_input_dim),
            nn.ReLU(inplace=True)
        )

        self.lstm = nn.LSTM(self.lstm_input_dim, self.num_lstm_hidden_units, num_layers=self.num_lstm_layers,
                            dropout=self.p_lstm_dropout, bidirectional=False)


        self.prob_layer = nn.Sequential(
            nn.Linear(self.num_lstm_hidden_units, 2)
        )

        #/ADDED
        # For now, only predicting Is input Cyrillic? if not then it must Latin (aka english)
        #cnn_output_h = int(self.input_line_height / 4)
        #self.linear = nn.Linear(256 * cnn_output_h, 2)

        # Finally, let's initialize parameters
        for param in self.parameters():
            torch.nn.init.uniform(param, -0.08, 0.08)

        total_params = 0
        for param in self.parameters():
            local_params = 1
            for d in param.size():
                local_params *= d
            total_params += local_params

        cnn_params = 0
        for param in self.cnn.parameters():
            local_params = 1
            for d in param.size():
                local_params *= d
            cnn_params += local_params

        lstm_params = 0
        for param in self.lstm.parameters():
            local_params = 1
            for d in param.size():
                local_params *= d
            lstm_params += local_params

        if self.verbose:
            logger.info("Total Model Params = %d" % total_params)
            logger.info("\tCNN Params = %d" % cnn_params)
            logger.info("\tLSTM Params = %d" % lstm_params)

            logger.info("Model looks like:")
            logger.info(repr(self))

        if torch.cuda.is_available() and self.gpu:
            self.cnn = self.cnn.cuda()
            #self.linear = self.linear.cuda()
            self.bridge_layer = self.bridge_layer.cuda()
            self.lstm = self.lstm.cuda()
            self.prob_layer = self.prob_layer.cuda()

            if self.multigpu:
                self.cnn = torch.nn.DataParallel(self.cnn)
        else:
            logger.info("Warning: Runnig model on CPU")

    # Consider: nn.Dropout2d
    def ConvBNReLU(self, nInputMaps, nOutputMaps):
        return [nn.Conv2d(nInputMaps, nOutputMaps, kernel_size=3, padding=1),
                nn.BatchNorm2d(nOutputMaps),
                nn.ReLU(inplace=True)]

    def cnn_output_num_channels(self):
        out_c = 0
        for module in self.cnn.modules():
            if isinstance(module, nn.Conv2d):
                out_c = module.out_channels
        return out_c

    def calculate_hw(self, module, out_h, out_w):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.MaxPool2d):
            if isinstance(module.padding, tuple):
                padding_y, padding_x = module.padding
            else:
                padding_y = padding_x = module.padding
            if isinstance(module.dilation, tuple):
                dilation_y, dilation_x = module.dilation
            else:
                dilation_y = dilation_x = module.dilation
            if isinstance(module.stride, tuple):
                stride_y, stride_x = module.stride
            else:
                stride_y = stride_x = module.stride
            if isinstance(module.kernel_size, tuple):
                kernel_size_y, kernel_size_x = module.kernel_size
            else:
                kernel_size_y = kernel_size_x = module.kernel_size

            out_h = math.floor((out_h + 2.0 * padding_y - dilation_y * (kernel_size_y - 1) - 1) / stride_y + 1)
            out_w = math.floor((out_w + 2.0 * padding_x - dilation_x * (kernel_size_x - 1) - 1) / stride_x + 1)
        elif isinstance(module, nn.FractionalMaxPool2d):
            if module.outh is not None:
                #out_h, out_w = module.output_size
                out_h, out_w = module.outh, module.outw
            else:
                #rh, rw = module.output_ratio
                rh, rw = module.rh, module.rw
                out_h, out_w = math.floor(out_h * rh), math.floor(out_w * rw)

        return out_h, out_w

    def cnn_input_size_to_output_size(self, in_size):
        out_h, out_w = in_size

        for module in self.cnn.modules():
            out_h, out_w = self.calculate_hw(module, out_h, out_w)

        return (out_h, out_w)

    def forward(self, x, actual_minibatch_widths):
        cnn_output = self.cnn(x)
        b, c, h, w = cnn_output.size()
       
        cnn_output = cnn_output.permute(3, 0, 1, 2).contiguous()

        lstm_input = self.bridge_layer(cnn_output.view(-1, c * h)).view(w, b, -1)

        # Try tensor.unfold(0, frame_size, step_size), e.g. with frame_size=2, step_size=1

        # Note: pack_padded_sequence assumes that minibatch elements are sorted by length
        #       i.e. minibatch_widths[0] is longest sequence and minibatch_widths[-1] is shortest sequence
        #       We assume that the data loader arranged input to conform to this constraint
        actual_cnn_output_widths = [self.cnn_input_size_to_output_size((self.input_line_height, width))[1] for width in actual_minibatch_widths.data]
        #try:
        #lstm_output = self.lstm(lstm_input)
        packed_lstm_input = rnn_pack(lstm_input, actual_cnn_output_widths)
       # except:
          #  print("minibatch widths: ")
         #   print (actual_minibatch_widths)
        #    print("actual cnn widths: ")
       #     print(actual_cnn_output_widths)
        packed_lstm_output, _ = self.lstm(packed_lstm_input)
        lstm_output, lstm_output_lengths = rnn_unpack(packed_lstm_output)
        #print("Lstm output: ",lstm_output.size())
        w = lstm_output.size(0)
       # print("Output lengths: ",len(lstm_output_lengths))
        prob_output = self.prob_layer(lstm_output.view(-1, lstm_output.size(2))).view(w, b, -1)

        #return prob_output
        return prob_output, lstm_output_lengths


        #cnn_output = cnn_output.permute(3, 0, 1, 2).contiguous()
        # Now collapse output to a single timestep using avg pooling
        #pooled_feat = torch.mean(cnn_output, dim=-1).cpu()
        #pooled_feat, _ = torch.max(cnn_output, dim=-1)
        # Now feed into linear classification layer
        #classification_output = self.linear(pooled_feat.view(b, -1))

        #return classification_output


