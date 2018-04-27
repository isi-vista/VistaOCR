import os
import argparse
import datetime

import torch
from models.cnnlstm import CnnOcrModel
from torch.autograd import Variable
from torchvision import transforms

from arabic import ArabicAlphabet
from datautils import GroupedSampler, SortByWidthCollater
from english import EnglishAlphabet
from iam import IAMDataset
from madcat import MadcatDataset
from textutils import *

import loggy
logger = loggy.setup_custom_logger('root', "test_madcat_iam.py")


def get_args():
    parser = argparse.ArgumentParser(description="OCR Testing Script")
    parser.add_argument("--line-height", type=int, default=30, help="Input image line height")
    parser.add_argument("--hpad", type=int, default=0,
                        help="Amount of horizontal padding to apply to left/right of input image (after resize)")
    parser.add_argument("--vpad", type=int, default=0,
                        help="Amount of vertical padding to apply to top/bottom of input image (after resize)")
    parser.add_argument("--image-list", type=str, required=True, help="Specify list of images to decode")
    parser.add_argument("--snapshot", type=str, required=True, help="Path to snapshot from which we should initialize model weights")
    parser.add_argument("--model", type=str, default='CnnOcrModel',
                        help="Model to use with the snapshot options")
    parser.add_argument("--lm-path", type=str, required=False, default="",
                        help="path to LM dir containing TLG.fst, words.txt & units.txt")
    parser.add_argument("--acoustic-weight", type=float, default=0.9, help="acoustic weight to initialize the LM with.")
    parser.add_argument("--tmpdir", type=str, default=os.environ.get('TMPDIR'), help="tmpdir to write the files for sclite script.")
    return parser.parse_args()


sys.stdout.flush()
args = get_args()
model_path = args.snapshot

line_img_transforms = transforms.Compose([
    transforms.Scale(new_h=args.line_height),
    transforms.InvertBlackWhite(),
    transforms.Pad(args.hpad, args.vpad),
    transforms.ToTensor(),
])

if args.lm_path != "":
    lm_units = os.path.join(args.lm_path, 'units.txt')
    lm_words = os.path.join(args.lm_path, 'words.txt')
    lm_wfst = os.path.join(args.lm_path, 'TLG.fst')
else:
    lm_units = ""

# need language arg (or read from model more likely!!!)
#alphabet = EnglishAlphabet(lm_units_path=lm_units)

if lm_units == "":
    lm_units = "/nas/home/srawls/madcat-units.txt"
alphabet = ArabicAlphabet(lm_units_path=lm_units)

if args.model == "CnnOcrModel":
    model = CnnOcrModel.FromSavedWeights(model_path, alphabet=alphabet)
else:
    raise TypeError("model not recognized.")
model.eval()

if args.lm_path != "":
    logger.info("About to init LM with acoustic_weight:%s" % args.acoustic_weight)
     model.init_lm(lm_wfst, lm_words, acoustic_weight=args.acoustic_weight)
     logger.info("Done init'ing LM")


hyp_output = []
hyp_lm_output = []

iteration = 0

total_model_time = None
total_word_count = 0

first_iteration = True

for input_tensor, target, input_widths, target_widths, metadata in test_dataloader:
    input_tensor = Variable(input_tensor.cuda(async=True), volatile=True)
    target = Variable(target, volatile=True)
    target_widths = Variable(target_widths, volatile=True)
    input_widths = Variable(input_widths, volatile=True)
    start = datetime.datetime.now()
    # model_output, model_output_actual_lengths, writer_id_output  = model(input_tensor, input_widths)
    model_output, model_output_actual_lengths = model(input_tensor, input_widths)
    elapsed_time = datetime.datetime.now() - start
    hyp_transcriptions = model.decode_without_lm(model_output, model_output_actual_lengths, uxxxx=True)
    if first_iteration:
        # Skip timing first iteration because it contains lots of overhead
        first_iteration = False
    else:
        if total_model_time is None:
            total_model_time = elapsed_time
        else:
            total_model_time += elapsed_time

    if args.dataset.upper() == "MADCAT":
        #hyp_transcriptions_lm = hyp_transcriptions
        hyp_transcriptions_lm = model.decode_with_lm(model_output, model_output_actual_lengths, uxxxx=True)
    else:
        hyp_transcriptions_lm = model.decode_with_lm(model_output, model_output_actual_lengths, uxxxx=True)

    batch_size = input_tensor.size(0)
    n_samples += 1

    cur_target_offset = 0
    batch_cer = 0
    batch_wer = 0
    batch_cer_lm = 0
    batch_wer_lm = 0

    #    writer_target = metadata['writer-ids']
    #    writer_id_predictions = writer_id_output.max(1)[1].type_as(writer_target)
    #    writer_correct = writer_id_predictions.data.eq(writer_target)
    #    if not hasattr(writer_correct, 'sum'):
    #        writer_correct = writer_correct.cpu()
    #    writer_correct = writer_correct.sum()
    #    writer_id_running_accuracy += (writer_correct/batch_size - writer_id_running_accuracy) / n_samples

    target_np = target.data.numpy()

    ref_transcriptions = []
    for i in range(len(hyp_transcriptions)):
        ref_transcription = form_target_transcription(
            target_np[cur_target_offset:(cur_target_offset + target_widths.data[i])], alphabet)
        ref_transcriptions.append(uxxxx_to_utf8(ref_transcription))
        cur_target_offset += target_widths.data[i]

        if not first_iteration:
            total_word_count += target_widths.data[i]

        cer, wer = compute_cer_wer(hyp_transcriptions[i], ref_transcription, alphabet)
        cer_lm, wer_lm = compute_cer_wer(hyp_transcriptions_lm[i], ref_transcription, alphabet)

        batch_cer += cer
        batch_wer += wer
        batch_cer_lm += cer_lm
        batch_wer_lm += wer_lm

        hyp_output.append((metadata['utt-ids'][i], hyp_transcriptions[i]))
        hyp_lm_output.append((metadata['utt-ids'][i], hyp_transcriptions_lm[i]))
        ref_output.append((metadata['utt-ids'][i], ref_transcription))

    cer_running_avg += (batch_cer / batch_size - cer_running_avg) / n_samples
    wer_running_avg += (batch_wer / batch_size - wer_running_avg) / n_samples
    cer_running_avg_lm += (batch_cer_lm / batch_size - cer_running_avg_lm) / n_samples
    wer_running_avg_lm += (batch_wer_lm / batch_size - wer_running_avg_lm) / n_samples

    logger.info("Iteration %d / %d" % (iteration, len(test_dataloader)))
    logger.info("\tcer_running_avg = %f, wer_running_avg = %f\tWriter Id Accuracy = %f" % (
    100 * cer_running_avg, 100 * wer_running_avg, 100 * writer_id_running_accuracy))
    logger.info(
    "\tWith LM: cer_running_avg = %f, wer_running_avg = %f" % (100 * cer_running_avg_lm, 100 * wer_running_avg_lm))
    if not total_model_time is None:
        logger.info("\t:Time: %s\tTotal Words = %d" % (pretty_print_timespan(total_model_time), total_word_count))
    logger.info("")
    iteration += 1

logger.info("Done.\n")
logger.info("Final CER = %f, WER = %f, Writer ID Accuracy = %f" % (
100 * cer_running_avg, 100 * wer_running_avg, 100 * writer_id_running_accuracy))
logger.info("With LM: Final CER = %f, WER = %f" % (100 * cer_running_avg_lm, 100 * wer_running_avg_lm))
logger.info("\t:Time: %s\tTotal Words = %d" % (pretty_print_timespan(total_model_time), total_word_count))

hyp_out_file = os.path.join(args.tmpdir, "%s-hyp-chars.txt" % args.dataset.lower())
hyp_lm_out_file = os.path.join(args.tmpdir, "%s-hyp-lm-chars.txt" % args.dataset.lower())
ref_out_file = os.path.join(args.tmpdir, "%s-ref-chars.txt" % args.dataset.lower())

with open(hyp_out_file, 'w') as fh:
    for uttid, hyp in hyp_output:
        fh.write("%s (%s)\n" % (hyp, uttid))

with open(hyp_lm_out_file, 'w') as fh:
    for uttid, hyp in hyp_lm_output:
        fh.write("%s (%s)\n" % (hyp, uttid))

with open(ref_out_file, 'w') as fh:
    for uttid, ref in ref_output:
        fh.write("%s (%s)\n" % (ref, uttid))
