import os
import argparse
import datetime

import torch
from models.cnnlstm import CnnOcrModel
from torch.autograd import Variable
import imagetransforms

from datautils import GroupedSampler, SortByWidthCollater
from ocr_dataset import OcrDataset
from textutils import *

import sys

def get_args():
    parser = argparse.ArgumentParser(description="OCR Training Script")
    parser.add_argument("--batch-size", type=int, default=64, help="SGD mini-batch size")

    parser.add_argument("--datadir", type=str, required=True, help="specify the location to data.")
    parser.add_argument("--outdir", type=str, required=True, help="specify the location to write output.")

    parser.add_argument("--num_data_threads", type=int, default=8, help="Number of background worker threads preparing/fetching data")

    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained OCR model used in decoding")

    parser.add_argument("--lm-path", type=str, required=False,
                        help="Path to trained language model for LM decoding")

    return parser.parse_args()


def main():
    args = get_args()

    model = CnnOcrModel.FromSavedWeights(args.model_path)
    model.eval()

    line_img_transforms = imagetransforms.Compose([
        imagetransforms.Scale(new_h=model.input_line_height),
        imagetransforms.InvertBlackWhite(),
        imagetransforms.ToTensor(),
    ])


    have_lm = (args.lm_path is not None) and (args.lm_path != "")

    if have_lm:
        lm_units = os.path.join(args.lm_path, 'units.txt')
        lm_words = os.path.join(args.lm_path, 'words.txt')
        lm_wfst = os.path.join(args.lm_path, 'TLG.fst')


    test_dataset = OcrDataset(args.datadir, "test", line_img_transforms)

    # Set seed for consistancy
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)


    if have_lm:
        model.init_lm(lm_wfst, lm_words, lm_units, acoustic_weight=0.8)


    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch_size,
                                                  num_workers=args.num_data_threads,
                                                  sampler=GroupedSampler(test_dataset, rand=False),
                                                  collate_fn=SortByWidthCollater,
                                                  pin_memory=True,
                                                  drop_last=False)


    hyp_output = []
    hyp_lm_output = []
    ref_output = []


    print("About to process test set. Total # iterations is %d." % len(test_dataloader))

    for idx, (input_tensor, target, input_widths, target_widths, metadata) in enumerate(test_dataloader):
        sys.stdout.write(".")
        sys.stdout.flush()

        # Wrap inputs in PyTorch Variable class
        input_tensor = Variable(input_tensor.cuda(async=True), volatile=True)
        target = Variable(target, volatile=True)
        target_widths = Variable(target_widths, volatile=True)
        input_widths = Variable(input_widths, volatile=True)

        # Call model
        model_output, model_output_actual_lengths = model(input_tensor, input_widths)

        # Do LM-free decoding
        hyp_transcriptions = model.decode_without_lm(model_output, model_output_actual_lengths, uxxxx=True)

        # Optionally, do LM decoding
        if have_lm:
            hyp_transcriptions_lm = model.decode_with_lm(model_output, model_output_actual_lengths, uxxxx=True)



        cur_target_offset = 0
        target_np = target.data.numpy()

        for i in range(len(hyp_transcriptions)):
            ref_transcription = form_target_transcription(
                target_np[cur_target_offset:(cur_target_offset + target_widths.data[i])], model.alphabet)
            cur_target_offset += target_widths.data[i]

            hyp_output.append((metadata['utt-ids'][i], hyp_transcriptions[i]))

            if have_lm:
                hyp_lm_output.append((metadata['utt-ids'][i], hyp_transcriptions_lm[i]))

            ref_output.append((metadata['utt-ids'][i], ref_transcription))


    hyp_out_file = os.path.join(args.outdir, "hyp-chars.txt")
    ref_out_file = os.path.join(args.outdir, "ref-chars.txt")

    if have_lm:
        hyp_lm_out_file = os.path.join(args.outdir, "hyp-lm-chars.txt")

    print("")
    print("Done. Now writing output files:")
    print("\t%s" % hyp_out_file)

    if have_lm:
        print("\t%s" % hyp_lm_out_file)

    print("\t%s" % ref_out_file)

    with open(hyp_out_file, 'w') as fh:
        for uttid, hyp in hyp_output:
            fh.write("%s (%s)\n" % (hyp, uttid))


    if have_lm:
        with open(hyp_lm_out_file, 'w') as fh:
            for uttid, hyp in hyp_lm_output:
                fh.write("%s (%s)\n" % (hyp, uttid))

    with open(ref_out_file, 'w') as fh:
        for uttid, ref in ref_output:
            fh.write("%s (%s)\n" % (ref, uttid))


if __name__ == "__main__":
    main()
