import os
import argparse
import datetime

import torch
from models.cnn_script_id import CnnScriptIdModel
from torch.autograd import Variable
import imagetransforms

from datautils import GroupedSampler, SortByWidthCollater_scriptid
from script_id_dataset import ScriptIdDataset
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

    model = CnnScriptIdModel.FromSavedWeights(args.model_path)
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


    num_input_channel = model.get_hyper_params().get('num_in_channels', 1)
    test_dataset = ScriptIdDataset(args.datadir, "test", line_img_transforms,numinputchannels=num_input_channel)
    print(test_dataset)
    # Set seed for consistancy
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)


    if have_lm:
        model.init_lm(lm_wfst, lm_words, lm_units, acoustic_weight=0.8)


    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch_size,
                                                  num_workers=args.num_data_threads,
                                                  sampler=GroupedSampler(test_dataset, rand=False),
                                                  collate_fn=SortByWidthCollater_scriptid,
                                                  pin_memory=True,
                                                  drop_last=False)


    hyp_output = []
    hyp_lm_output = []
    ref_output = []
    font_hyp_output = []
    font_ref_output = []
    unusualfontpath = "/nas/home/jschmidt/SynthText/data/fonts/unusualfonts.txt"
    unusualfonts = [line.strip().lower() for line in open(unusualfontpath)]
    print(unusualfonts)
    #print(unusualfonts)
    print("About to process test set. Total # iterations is %d." % len(test_dataloader))

    for idx, (input_tensor, target, metadata) in enumerate(test_dataloader):
        sys.stdout.write(".")
        sys.stdout.flush()
        #print("Target: ",target)
        #print("Metadata: ", metadata)
        # Wrap inputs in PyTorch Variable class
        input_tensor = Variable(input_tensor.cuda(async=True), volatile=True)
        target = Variable(target, volatile=True)
        # Call model
        #model_output, model_output_actual_lengths = model(input_tensor, input_widths)
        model_output = model(input_tensor)
        print(model_output)
        # Do LM-free decoding
        #hyp_transcriptions = model.decode_without_lm(model_output, model_output_actual_lengths, uxxxx=True)
        # Optionally, do LM decoding
        maxout, maxindex = torch.max(model_output,1)
        print(maxindex)

        print("Predictions: " , maxindex)
        cur_target_offset = 0
        target_np = target.data.numpy()

        for i in range(len(maxindex)):
            ref_transcription = metadata['trans_raw'][i]
            label = metadata['label'][i]
            hyp = maxindex[i].data[0]
            #print("R1: ", ref_transcription)
            #ref_transcription = form_target_transcription(
             #  target_np[cur_target_offset:(cur_target_offset + target_widths.data[i])], model.alphabet)
             #   target_np[cur_target_offset:(cur_target_offset + target_widths.data[i])], test_dataset.alphabet)
             
            print("H: ", hyp, "  R: ", label)

            #cur_target_offset += target_widths.data[i]

            hyp_output.append((metadata['utt-ids'][i], hyp,metadata['font_family'][i]))

            ref_output.append((metadata['utt-ids'][i], label,metadata['font_family'][i]))
            font_family = metadata['font_family'][i].strip().lower()
            #print("Font:",font_family)
            if(font_family in unusualfonts):
                font_type = "Unusual"
            else:
                font_type = "Standard"
            font_type_id = font_type + "-" + metadata['utt-ids'][i]
            font_hyp_output.append((font_type_id, hyp))
            font_ref_output.append((font_type_id, label))



    hyp_out_file = os.path.join(args.outdir, "hyp-chars.txt")
    ref_out_file = os.path.join(args.outdir, "ref-chars.txt")
    font_hyp_out_file = os.path.join(args.outdir, "font-hyp-chars.txt")
    font_ref_out_file = os.path.join(args.outdir, "font-ref-chars.txt")


    if have_lm:
        hyp_lm_out_file = os.path.join(args.outdir, "hyp-lm-chars.txt")
        

    print("")
    print("Done. Now writing output files:")
    print("\t%s" % hyp_out_file)

    if have_lm:
        print("\t%s" % hyp_lm_out_file)

    print("\t%s" % ref_out_file)

    with open(hyp_out_file, 'w') as fh:
        for uttid, hyp, font_family in hyp_output:
            #fh.write("%s (%s)\n" % (hyp, uttid))
            newid = font_family + "-" + uttid
            fh.write("%s (%s) \n" % (hyp, newid))


    if have_lm:
        with open(hyp_lm_out_file, 'w') as fh:
            for uttid, hyp in hyp_lm_output:
                fh.write("%s (%s)\n" % (hyp, uttid))

    with open(font_ref_out_file, 'w') as fh:
        for newid, ref in font_ref_output:
            fh.write("%s (%s)\n" % (ref, newid))

    with open(font_hyp_out_file, 'w') as fh:
        for newid, hyp in font_hyp_output:
            fh.write("%s (%s) \n" % (hyp, newid))


#    if have_lm:
 #       with open(hyp_lm_out_file, 'w') as fh:
  #          for uttid, hyp in hyp_lm_output:
   #             fh.write("%s (%s)\n" % (hyp, uttid))

    with open(ref_out_file, 'w') as fh:
        for uttid, ref, font_family in ref_output:
            #fh.write("%s (%s)\n" % (ref, uttid))
            newid = font_family + "-" + uttid
            fh.write("%s (%s)\n" % (ref, newid))

if __name__ == "__main__":
    main()
