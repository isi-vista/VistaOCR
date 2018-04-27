import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import make_grid
from torchvision import transforms
import imagetransforms

import numpy as np
import random
import scipy.misc
import math
from warpctc_pytorch import CTCLoss
from arabic import ArabicAlphabet
from english import EnglishAlphabet
import sys

import datetime
import shutil

from madcat import MadcatDataset
from iam import IAMDataset
from datautils import GroupedSampler, SortByWidthCollater
import models.cnnlstm as MODELS
from models.cnnlstm import CnnOcrModel
from textutils import *
import cv2

def load_model(fpath, fdim=128, hu=256, gpu=False):

    mweights = torch.load(fpath)

    # For now hardcode these params
    # Later need to read them from serialized model file
    line_height = 30
    h_pad = 0
    v_pad = 0

    alphabet = EnglishAlphabet()
    #alphabet = ArabicAlphabet()

    model = CnnOcrModel(
        num_in_channels = 1,
        input_line_height = line_height + 2*v_pad,
        lstm_input_dim = fdim,
        num_lstm_layers = 3,
        num_lstm_hidden_units = hu,
        p_lstm_dropout = 0.5,
        alphabet = alphabet,
        multigpu = True,
        verbose = False,
        gpu=gpu)

    model.load_state_dict(mweights['state_dict'])
    model.eval()

    return model

def convert_to_cpu(mpath_in, mpath_out):
    # (1) Move weights from GPU to CPU
    mweights = torch.load(mpath_in, map_location={'cuda:0':'cpu'})

    # (2) Model doesn't use DataParallel wrapper on CPU, so need to munge state dict names
    keys = list(mweights['state_dict'].keys())
    for key in keys:
        if 'cnn.module' in key:
            new_key = key.replace('cnn.module', 'cnn')
            mweights['state_dict'][new_key] = mweights['state_dict'][key]
            del mweights['state_dict'][key]

    # All done, save modified model weights
    torch.save(mweights, mpath_out)


IamTestDataset = None
def get_random_iam_test_sample(lh=30):
    global IamTestDataset
    if IamTestDataset is None:
        # hardcoded for now(!)
        line_height = lh
        h_pad = 0
        v_pad = 0
        line_img_transforms = imagetransforms.Compose([
            imagetransforms.Scale(new_h = line_height),
            imagetransforms.InvertBlackWhite(),
            imagetransforms.Pad(h_pad, v_pad),
            imagetransforms.ToTensor(),
        ])
        IamTestDataset = IAMDataset("/nfs/isicvlnas01/users/srawls/ocr-dev/data/iam/", "test", EnglishAlphabet(lm_units_path="/nfs/isicvlnas01/users/jmathai//experiments/lm_grid_search/iam-grid-data/IAM-LM-4-kndiscount-interpolate-0.9/IAM-LM/units.txt"), line_height, line_img_transforms)

    return IamTestDataset[ random.randint(0, len(IamTestDataset)-1) ]



MadcatTestDataset = None
def get_random_madcat_test_sample(lh=30):
    global MadcatTestDataset
    if MadcatTestDataset is None:
        # hardcoded for now(!)
        line_height = lh
        h_pad = 0
        v_pad = 0
        line_img_transforms = imagetransforms.Compose([
            imagetransforms.Scale(new_h = line_height),
            imagetransforms.InvertBlackWhite(),
            imagetransforms.Pad(h_pad, v_pad),
            imagetransforms.ToTensor(),
        ])
        MadcatTestDataset = MadcatDataset("/lfs2/srawls/madcat", "test", ArabicAlphabet(), line_height, line_img_transforms)

    return MadcatTestDataset[ random.randint(0, len(MadcatTestDataset)-1) ]

    
def decode_single_sample(model, input_tensor, uxxxx=False):
    # Add a batch dimension
    model_input = input_tensor.view(1, input_tensor.size(0), input_tensor.size(1), input_tensor.size(2))
    input_widths = torch.autograd.Variable(torch.IntTensor( [model_input.size(3)] ))

    # Move to GPU if using cuda
    if torch.cuda.is_available() and model.gpu:
        model_input = model_input.cuda()

    # Wrap in a Torch Variable instance, because model expects that
    model_input = torch.autograd.Variable(model_input)

    model_output, model_output_actual_lengths = model(model_input, input_widths)

    hyp = model.decode_without_lm(model_output, model_output_actual_lengths, uxxxx=uxxxx)

    return model_output, hyp[0]

def decode_single_sample_withlm(model, input_tensor, uxxxx=False):
    # Add a batch dimension
    model_input = input_tensor.view(1, input_tensor.size(0), input_tensor.size(1), input_tensor.size(2))
    input_widths = torch.autograd.Variable(torch.IntTensor( [model_input.size(3)] ))

    # Move to GPU if using cuda
    if torch.cuda.is_available() and model.gpu:
        model_input = model_input.cuda()

    # Wrap in a Torch Variable instance, because model expects that
    model_input = torch.autograd.Variable(model_input)

    model_output, model_output_actual_lengths = model(model_input, input_widths)

    hyp = model.decode_with_lm(model_output, model_output_actual_lengths, uxxxx=uxxxx)

    return model_output, hyp[0]


def decode_single_sample_return_hidden(model, input_tensor, gpu=False):
    # Add a batch dimension
    model_input = input_tensor.view(1, input_tensor.size(0), input_tensor.size(1), input_tensor.size(2))
    input_widths = torch.autograd.Variable(torch.IntTensor( [model_input.size(3)] ))

    # Move to GPU if using cuda
    if torch.cuda.is_available() and gpu:
        model_input = model_input.cuda()

    # Wrap in a Torch Variable instance, because model expects that
    model_input = torch.autograd.Variable(model_input)

    model_output, model_output_actual_lengths, hidden = model.forward_return_hidden(model_input, input_widths)

    hyp = model.decode_without_lm(model_output, model_output_actual_lengths, uxxxx=False)

    return model_output, hyp[0], hidden

