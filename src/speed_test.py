import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import imagetransforms

from madcat import MadcatDataset
from iam import IAMDataset
from datautils import GroupedSampler, SortByWidthCollater
from models.cnnlstm import CnnOcrModel

from warpctc_pytorch import CTCLoss
import time
import numpy as np

batchsize = 64
height=30

print("batch size = %d" % batchsize)
print("height = %d" % height)

model = CnnOcrModel(
    num_in_channels=1,
    input_line_height=height,
    lstm_input_dim=128,
    num_lstm_layers=3,
    num_lstm_hidden_units=512,
    p_lstm_dropout=0.5,
    alphabet=range(120),
    multigpu=False)

model.train()

criterion = CTCLoss().cuda()


# Setup fake constant target
target = torch.autograd.Variable(torch.IntTensor(batchsize*5))
for idx in range(target.size(0)):
    target[idx] = 32
target_widths = torch.autograd.Variable(torch.IntTensor([5]*batchsize))


print("torch.backends.cudnn.enabled = %s" % torch.backends.cudnn.enabled)
torch.backends.cudnn.benchmark = False

print("torch.backends.cudnn.benchmark = %s" % torch.backends.cudnn.benchmark)
print("")

for width in [100,150,200,250,300,350,400,450,500,550,600]:
    times = []
    for i in range(20):
        start = time.time()

        input_tensor = torch.autograd.Variable(torch.randn( batchsize,1,height,width ).cuda())
        input_widths = torch.autograd.Variable(torch.LongTensor( [width]*batchsize ))
        model_output, model_output_actual_lengths = model(input_tensor, input_widths)
        loss = criterion(model_output, target, model_output_actual_lengths, target_widths)
        loss.backward()

        torch.cuda.synchronize()

        end = time.time()

        elapsed = (end-start)
        times.append(elapsed)

    avg_time = np.average(times[5:])
    print("Width=%d, Avg Time = %0.2f" % (width, avg_time))
