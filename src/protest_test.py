import os
import argparse
import datetime

import torch
from models.cnnlstm import CnnOcrModel
from torch.autograd import Variable
import imagetransforms

from datautils import GroupedSampler, SortByWidthCollater
from iam import IAMDataset
from rimes import RimesDataset
from madcat import MadcatDataset
from textutils import *

import conf_utils

from english import EnglishAlphabet
from french import FrenchAlphabet

import sys

import cv2

iam = True

if iam:
    model_path = "/nas/home/srawls/ocr/experiments/iam-baseline-best_model.pth"
    lm_path = "/nfs/isicvlnas01/users/jmathai/experiments/iam_lm_augment_more_data/IAM-LM/"

line_height = 120

line_img_transforms = imagetransforms.Compose([
    imagetransforms.Scale(new_h=line_height),
    imagetransforms.InvertBlackWhite(),
    imagetransforms.ToTensor(),
])

lm_units = os.path.join(lm_path, 'units.txt')
lm_words = os.path.join(lm_path, 'words.txt')
lm_wfst = os.path.join(lm_path, 'TLG.fst')


# Set seed for consistancy
torch.manual_seed(7)
torch.cuda.manual_seed_all(7)


model = CnnOcrModel.FromSavedWeights(model_path)
model.eval()

model.init_lm(lm_wfst, lm_words, acoustic_weight=0.8)

print("Starting on data")

#img = '/nas/home/srawls/aida-ocr/rex-text-detection/text-regions/004291text-region-Idx6-Pr0.781796395778656.jpg'
#for img_file in [img]:
base_dir = '/nas/home/srawls/aida-ocr/rex-text-detection/text-regions/'

bad_imgs = ['003991text-region-Idx57-Pr0.7098091840744019.jpg']

hyp_out_file = os.path.join(os.environ["TMPDIR"], "hyp-chars.txt")
hyp_lm_out_file = os.path.join(os.environ["TMPDIR"], "hyp-lm-chars.txt")

with open(hyp_out_file, 'w') as fh_hyp, open(hyp_lm_out_file, 'w') as fh_hyp_lm:
    for idx, f in enumerate(os.listdir(base_dir)):
        if f in bad_imgs:
            continue

        print("[%d] Operating on %s" % (idx,f))
        sys.stdout.flush()

        img_file = base_dir + "/" + f
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        # First do binarization
        th,bin_img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        input_tensor = line_img_transforms(bin_img)

        input_widths = torch.LongTensor([input_tensor.size(2)])
        input_tensor.unsqueeze_(0)

        input_tensor = Variable(input_tensor.cuda(async=True), volatile=True)
        input_widths = Variable(input_widths, volatile=True)

        model_output, conf_output, model_output_actual_lengths = model(input_tensor, input_widths)

        hyp_transcriptions = model.decode_without_lm(model_output, model_output_actual_lengths, uxxxx=False)
        hyp_transcriptions_lm = model.decode_with_lm(model_output, model_output_actual_lengths, uxxxx=False)

        for i in range(len(hyp_transcriptions)):
            fh_hyp.write("%s\n" % hyp_transcriptions[i])
            fh_hyp.flush()
            fh_hyp_lm.write("%s\n" % hyp_transcriptions_lm[i])
            fh_hyp_lm.flush()



print("")

print("Done.")
