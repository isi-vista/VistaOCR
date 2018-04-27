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

from sklearn.metrics import roc_auc_score


def compute_pr(y_gt, y_pred, thresh):
    tp = fp = tn = fn = 0

    for gt,hyp in zip(y_gt, y_pred):
        if gt == 1:
            if hyp < thresh:
                fn += 1
            else:
                tp += 1
        else:
            if hyp < thresh:
                tn += 1
            else:
                fp += 1

    precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0 if (tp + fn) == 0 else tp / (tp + fn)

    return precision,recall

rimes = iam = False

if sys.argv[1] == "rimes":
    rimes = True
if sys.argv[1] == "iam":
    iam = True

if sys.argv[2] == "PR":
    only_pr = True
else:
    only_pr = False


if iam:
    model_path = "/nas/home/srawls/ocr/experiments/iam-conf1-cur_snapshot.pth"
    lm_path = "/nfs/isicvlnas01/users/jmathai/experiments/iam_lm_augment_more_data/IAM-LM/"
    datadir = os.environ["TMPDIR"] + "/iam"
if rimes:
    #model_path = "/nas/home/srawls/ocr/experiments/rimes-mos-best_model.pth"
    model_path = "/nfs/isicvlnas01/users/jmathai/models/rimes_pytorch_models/model_v04.pth"
    lm_path = "/nfs/isicvlnas01/users/jmathai/models/rimes_kenlm_models/rimes_6_interpolate_20k_wiki_giga/"
    datadir = os.environ["TMPDIR"] + "/rimes"

line_height = 120

line_img_transforms = imagetransforms.Compose([
    imagetransforms.Scale(new_h=line_height),
    imagetransforms.InvertBlackWhite(),
    imagetransforms.ToTensor(),
])

lm_units = os.path.join(lm_path, 'units.txt')
lm_words = os.path.join(lm_path, 'words.txt')
lm_wfst = os.path.join(lm_path, 'TLG.fst')


if iam:
    alphabet = EnglishAlphabet(lm_units_path=lm_units)
    test_dataset = IAMDataset(datadir, "test", alphabet, line_height, line_img_transforms)
if rimes:
    test_dataset = RimesDataset(datadir, "test", line_height, line_img_transforms)

# Set seed for consistancy
torch.manual_seed(7)
torch.cuda.manual_seed_all(7)


model = CnnOcrModel.FromSavedWeights(model_path)
model.eval()

if not only_pr:
    model.init_lm(lm_wfst, lm_words, acoustic_weight=0.8)

test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=32,
                                              num_workers=8,
                                              sampler=GroupedSampler(test_dataset, rand=False),
                                              collate_fn=SortByWidthCollater,
                                              pin_memory=True,
                                              drop_last=False)

hyp_output = []
hyp_lm_output = []
hyp_lm_conf_output = []
ref_output = []

y_gt = []
y_pred = []

print("Starting on %d iterations over test set." % len(test_dataloader))
for idx, (input_tensor, target, input_widths, target_widths, metadata) in enumerate(test_dataloader):

#    if idx >= 26:
#        break # Early exit for quicker results (on smaller set!)

    sys.stdout.write(".")
    sys.stdout.flush()

    input_tensor = Variable(input_tensor.cuda(async=True), volatile=True)
    target = Variable(target, volatile=True)
    target_widths = Variable(target_widths, volatile=True)
    input_widths = Variable(input_widths, volatile=True)

    model_output, conf_output, model_output_actual_lengths = model(input_tensor, input_widths)

    hyp_transcriptions = model.decode_without_lm(model_output, model_output_actual_lengths, uxxxx=True)

    if not only_pr:
        hyp_transcriptions_lm = model.decode_with_lm(model_output, model_output_actual_lengths, uxxxx=True)
    else:
        hyp_transcriptions_lm = hyp_transcriptions

    # Now get hyps w/ conf scores
    #prob_success = torch.nn.functional.softmax(conf_output, dim=2).cpu().data.numpy()[:,:,0]

    #prob_success = torch.max(torch.nn.functional.softmax(model_output, dim=2), dim=2)[0].cpu().data.numpy()

    # Tmp over-ride w/ ground truth
    #conf_gt = conf_utils.form_confidence_gt(model_output, model_output_actual_lengths, target.data.numpy(), target_widths.data.numpy())
    #conf_gt = conf_gt.data

    # Want to keep track of, let's say, all char-frames
    #for t in range(conf_gt.size(0)):
    #    for b in range(conf_gt.size(1)):
    #        if conf_gt[t,b] == 2:
    #            continue
    #
    #        y_gt.append(conf_gt[t,b])
    #        y_pred.append(1-prob_success[t,b])


#    print("DBG:")
#    print("At Threshold 0.5,  Precision = %f, Recall = %f" % compute_pr(y_gt, y_pred, 0.5))
#    print("At Threshold 0.7,  Precision = %f, Recall = %f" % compute_pr(y_gt, y_pred, 0.7))
#    print("At Threshold 0.9,  Precision = %f, Recall = %f" % compute_pr(y_gt, y_pred, 0.9))
#    print("")


#    err_cnt = 0
#    for t in range(conf_gt.size(0)):
#        for b in range(conf_gt.size(1)):
#            if conf_gt[t,b] == 1: #error case
#                prob_success[t,b] = 0
#                err_cnt += 1
#            else:
#                # Not an error
#                prob_success[t,b] = 1
#
#    print("dbg: err cnt = %d" % err_cnt)

#    if not only_pr:
    if False:
        hyp_transcriptions_lm_conf = model.decode_with_lm_with_conf(model_output, model_output_actual_lengths,
                                                                    prob_success, uxxxx=True,
                                                                    high_thresh=0.99,
                                                                    low_thresh=0.3,
                                                                    high_aw=1.0,
                                                                    mid_aw=0.7,
                                                                    low_aw=0.5)
    else:
        hyp_transcriptions_lm_conf = hyp_transcriptions

#    hyp_transcriptions_lm_conf = model.decode_with_lm(model_output, model_output_actual_lengths, uxxxx=True, pmod=True)



    cur_target_offset = 0
    target_np = target.data.numpy()

    for i in range(len(hyp_transcriptions)):
        ref_transcription = form_target_transcription(
            target_np[cur_target_offset:(cur_target_offset + target_widths.data[i])], model.alphabet)
        cur_target_offset += target_widths.data[i]

        hyp_output.append((metadata['utt-ids'][i], hyp_transcriptions[i]))
        hyp_lm_output.append((metadata['utt-ids'][i], hyp_transcriptions_lm[i]))
        hyp_lm_conf_output.append((metadata['utt-ids'][i], hyp_transcriptions_lm_conf[i]))

        ref_output.append((metadata['utt-ids'][i], ref_transcription))


print("")

#with open("/tmp/roc.txt", 'w') as fh:
#    for h,g in zip(y_pred, y_gt):
#        fh.write("%f,%f\n" % (h,g))
#        
#
#
#auc = roc_auc_score(y_gt, y_pred)
#print("AUC Score of error classifications is: %f" % auc)
#print("At Threshold 0.1,  Precision = %f, Recall = %f" % compute_pr(y_gt, y_pred, 0.1))
#print("At Threshold 0.3,  Precision = %f, Recall = %f" % compute_pr(y_gt, y_pred, 0.3))
#print("At Threshold 0.5,  Precision = %f, Recall = %f" % compute_pr(y_gt, y_pred, 0.5))
#print("At Threshold 0.7,  Precision = %f, Recall = %f" % compute_pr(y_gt, y_pred, 0.7))
#print("At Threshold 0.9,  Precision = %f, Recall = %f" % compute_pr(y_gt, y_pred, 0.9))

if iam:
    hyp_out_file = os.path.join(os.environ["TMPDIR"], "iam-hyp-chars.txt")
    hyp_lm_out_file = os.path.join(os.environ["TMPDIR"], "iam-hyp-lm-chars.txt")
    hyp_lm_conf_out_file = os.path.join(os.environ["TMPDIR"], "iam-hyp-lm-conf-chars.txt")
    ref_out_file = os.path.join(os.environ["TMPDIR"], "iam-ref-chars.txt")
if rimes:
    hyp_out_file = os.path.join(os.environ["TMPDIR"], "rimes-hyp-chars.txt")
    hyp_lm_out_file = os.path.join(os.environ["TMPDIR"], "rimes-hyp-lm-chars.txt")
    hyp_lm_conf_out_file = os.path.join(os.environ["TMPDIR"], "rimes-hyp-lm-conf-chars.txt")
    ref_out_file = os.path.join(os.environ["TMPDIR"], "rimes-ref-chars.txt")


print("Done. Now writing output files:")
print("\t%s" % hyp_out_file)
print("\t%s" % hyp_lm_out_file)
print("\t%s" % hyp_lm_conf_out_file)
print("\t%s" % ref_out_file)


with open(hyp_out_file, 'w') as fh:
    for uttid, hyp in hyp_output:
        fh.write("%s (%s)\n" % (hyp, uttid))

with open(hyp_lm_out_file, 'w') as fh:
    for uttid, hyp in hyp_lm_output:
        fh.write("%s (%s)\n" % (hyp, uttid))

with open(hyp_lm_conf_out_file, 'w') as fh:
    for uttid, hyp in hyp_lm_conf_output:
        fh.write("%s (%s)\n" % (hyp, uttid))

with open(ref_out_file, 'w') as fh:
    for uttid, ref in ref_output:
        fh.write("%s (%s)\n" % (ref, uttid))
