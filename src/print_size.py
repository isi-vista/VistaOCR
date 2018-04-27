import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import imagetransforms
import numpy as np

from warpctc_pytorch import CTCLoss
from arabic import ArabicAlphabet
from english import EnglishAlphabet

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import datetime
import time
import shutil

from madcat import MadcatDataset
from iam import IAMDataset
from datautils import GroupedSampler, SortByWidthCollater
from models.cnnlstm import CnnOcrModel
from textutils import *
import argparse

from lr_scheduler import ReduceLROnPlateau

import GPUtil

import loggy
logger = loggy.setup_custom_logger('root', "train_cnn_lstm.py")

#alphabet = EnglishAlphabet("/nfs/isicvlnas01/users/jmathai/Experiments/iam_lm_augment_more_data/IAM-LM/units.txt")
alphabet = EnglishAlphabet("/nfs/isicvlnas01/users/jmathai//experiments/lm_grid_search/iam-grid-data/IAM-LM-4-kndiscount-interpolate-0.9/IAM-LM/units.txt")
LINEH = 240

model = CnnOcrModel(
    verbose=True,
    num_in_channels=1,
    input_line_height=LINEH,
    lstm_input_dim=128,
    num_lstm_layers=3,
    num_lstm_hidden_units=512,
    p_lstm_dropout=0.5,
    alphabet=alphabet,
    multigpu=True)

print("")
print("")
print("")
torch.cuda.empty_cache()
GPUtil.showUtilization()

torch.backends.cudnn.benchmark = True


# Setup fake constant target
batchsize=4
target = torch.autograd.Variable(torch.IntTensor(batchsize*5))
for idx in range(target.size(0)):
    target[idx] = 32
target_widths = torch.autograd.Variable(torch.IntTensor([5]*batchsize))

rapid_times = []
cnn_times = []
bridge_times = []
lstm_times = []
prob_times = []

for _ in range(30):
    #width = int(600 * (LINEH/30))
    width = int(975 * (LINEH/30))
    input_tensor = torch.randn(batchsize, 1, LINEH, width)
    input_widths = torch.LongTensor( [width]*batchsize )

    ctc_loss = CTCLoss().cuda()

    input_tensor = Variable(input_tensor.cuda(async=True))
    input_widths = Variable(input_widths)

    torch.cuda.synchronize()
    tick = time.time()
    model_output, model_output_actual_lengths, rapid_time, cnn_time, bridge_time, lstm_time, prob_time = model(input_tensor, input_widths)

    rapid_times.append(rapid_time)
    cnn_times.append(cnn_time)
    bridge_times.append(bridge_time)
    lstm_times.append(lstm_time)
    prob_times.append(prob_time)

    torch.cuda.synchronize()
    tock = time.time()

    print("Forward Pass: %f" % (tock-tick))

    torch.cuda.synchronize()
    tick = time.time()
    loss = ctc_loss(model_output, target, model_output_actual_lengths, target_widths)

    torch.cuda.synchronize()
    tock = time.time()

    print("CTC Loss: %f" % (tock-tick))

    torch.cuda.synchronize()
    tick = time.time()
    loss.backward()

    torch.cuda.synchronize()
    tock = time.time()

    print("Backward Pass: %f" % (tock-tick))


print("")
#torch.cuda.empty_cache()
print("Done.")


print("")
rapid_time = np.mean(rapid_times[5:])
cnn_time = np.mean(cnn_times[5:])
bridge_time = np.mean(bridge_times[5:])
lstm_time = np.mean(lstm_times[5:])
prob_time = np.mean(prob_times[5:])

total_time = rapid_time + cnn_time + bridge_time + lstm_time + prob_time
print("Average...")
print("Rapid DS Forward Pass: %f (%f%%)" % (100*rapid_time, 100*rapid_time/total_time))
print("CNN Forward Pass: %f (%f%%)" % (100*cnn_time, 100*cnn_time/total_time))
print("Bridge-layer Forward Pass: %f (%f%%)" % (100*bridge_time, 100*bridge_time/total_time))
print("LSTM Forward Pass: %f (%f%%)" % (100*lstm_time, 100*lstm_time/total_time))
print("Prob layer Forward Pass: %f  (%f%%)" % (100*prob_time, 100*prob_time/total_time))
