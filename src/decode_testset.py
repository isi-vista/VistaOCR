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

from decoder import ArgmaxDecoder, LmDecoder
import multiprocessing
import concurrent
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack

import sys

def get_args():
    parser = argparse.ArgumentParser(description="OCR Training Script")
    parser.add_argument("--batch-size", type=int, default=64, help="SGD mini-batch size")

    parser.add_argument("--datadir", type=str, required=True, help="specify the location to data.")
    parser.add_argument("--outdir", type=str, required=True, help="specify the location to write output.")

    parser.add_argument("--num_data_threads", type=int, default=8, help="Number of background worker threads preparing/fetching data")

    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained OCR model used in decoding")

    parser.add_argument("--cvtGray", default=False, action='store_true', help="Set if need to convert color images to grayscale") 

    parser.add_argument("--lm-path", type=str, required=False,
                        help="Path to trained language model for LM decoding")

    return parser.parse_args()


def main():
    args = get_args()

    model = CnnOcrModel.FromSavedWeights(args.model_path)
    model.eval()

    line_img_transforms = []

    if args.cvtGray:
        line_img_transforms.append(imagetransforms.ConvertGray())

    line_img_transforms.append(imagetransforms.Scale(new_h=model.input_line_height))

    # Only do for grayscale
    if model.num_in_channels == 1:
        line_img_transforms.append(imagetransforms.InvertBlackWhite())

    # For right-to-left languages
#    if model.rtl:
#        line_img_transforms.append(imagetransforms.HorizontalFlip())

    line_img_transforms.append(imagetransforms.ToTensor())

    line_img_transforms = imagetransforms.Compose(line_img_transforms)

    test_dataset = OcrDataset(args.datadir, "test", line_img_transforms, max_allowed_width=1e5)

    # Set seed for consistancy
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)


    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch_size,
                                                  num_workers=args.num_data_threads,
                                                  sampler=GroupedSampler(test_dataset, rand=False),
                                                  collate_fn=SortByWidthCollater,
                                                  pin_memory=True,
                                                  drop_last=False)


    print("About to process test set. Total # iterations is %d." % len(test_dataloader))

    # Setup seperate process + queue for handling CPU-portion of decoding
    input_queue = multiprocessing.Queue()
    decoding_p = multiprocessing.Process(target=decode_thread, args=(input_queue, args.outdir, model.alphabet, args.lm_path))
    decoding_p.start()

    # No need for backprop during validation test
    start_time = datetime.datetime.now()
    with torch.no_grad():
        for idx, (input_tensor, target, input_widths, target_widths, metadata) in enumerate(test_dataloader):
            # Wrap inputs in PyTorch Variable class
            input_tensor = input_tensor.cuda(async=True)

            # Call model
            model_output, model_output_actual_lengths = model(input_tensor, input_widths)

            # Put model output on the queue for background process to decode
            input_queue.put( (model_output.cpu(), model_output_actual_lengths, metadata) )

    # Now we just need to wait for decode thread to finish
    input_queue.put( None )
    input_queue.close()
    decoding_p.join()

    end_time = datetime.datetime.now()

    print("Decoding took %f seconds" % (end_time - start_time).total_seconds())




def decode_thread(input_queue, outdir, alphabet, lm_path):
    # Setup LM args (if any)
    have_lm = (lm_path is not None) and (lm_path != "")

    n_workers = 10

    if have_lm:
        lm_units = os.path.join(lm_path, 'units.txt')
        lm_words = os.path.join(lm_path, 'words.txt')
        lm_wfst = os.path.join(lm_path, 'TLG.fst')


    # First initialize decoders
    argmax_decoder = ArgmaxDecoder(alphabet)
    if have_lm:
        lm_decoder = LmDecoder(alphabet, lm_wfst, lm_words, lm_units, acoustic_weight=0.8)

    # Then start waiting on input
    hyp_out_file = os.path.join(outdir, "hyp-chars.txt")
    if have_lm:
        hyp_lm_out_file = os.path.join(outdir, "hyp-lm-chars.txt")

    with ExitStack() as exit_stack:
        fh_hyp_out = exit_stack.enter_context(open(hyp_out_file, 'w'))
        fh_hyp_out_utf8 = exit_stack.enter_context(open(hyp_out_file + ".utf8", 'w', encoding="utf-8"))

        if have_lm:
            fh_hyp_lm_out = exit_stack.enter_context(open(hyp_lm_out_file, 'w'))
            tp_executor = exit_stack.enter_context(ThreadPoolExecutor(max_workers=n_workers))

        # Unknown mem leak caused by executor
        # Tmp workaround: every so often flush out all current tasks; shutdown current executor, then start up another
        batch_cnt=0
        while True:
            next_bacth = input_queue.get()
            if next_bacth is None:
                break

            if have_lm:
                batch_cnt+=1
                if batch_cnt >= 10:
                    # No need to wait for shutdown to finish, just slows us down
                    # Does mean that some slow-running decodes could cause us to go over the number of max workers
                    # For now just live with it
                    tp_executor.shutdown(wait=False)
                    tp_executor = exit_stack.enter_context(ThreadPoolExecutor(max_workers=n_workers))
                    batch_cnt = 0

            model_output, model_output_actual_lengths, metadata = next_bacth

            # Do LM-free decoding
            hyp_transcriptions = argmax_decoder.decode(model_output, model_output_actual_lengths, uxxxx=False)

            # Optionally, do LM decoding
            if have_lm:
                hyp_transcriptions_lm_futures = lm_decoder.decode(tp_executor, model_output, model_output_actual_lengths, metadata['utt-ids'], uxxxx=False)

                def get_result_from_future(future):
                    raw_result = future.result()
                    # 1) reverse result
                    #visual = ''.join([char for char in reversed(raw_result)])
                    visual = raw_result
                    # 2) Convert from visual to logical
                    logical = utf8_visual_to_logical(visual)
                    # 3) Convert to uxxxx format
                    return logical
                    #return utf8_to_uxxxx(logical)

                for i, future in enumerate(hyp_transcriptions_lm_futures):
                    uttid = metadata['utt-ids'][i]
                    future.add_done_callback(lambda f,uttid=uttid: fh_hyp_lm_out.write("%s (%s)\n" % (get_result_from_future(f), uttid)))


            for i in range(len(hyp_transcriptions)):
                hyp_utf8 = utf8_visual_to_logical(hyp_transcriptions[i])
                #hyp_utf8 = hyp_transcriptions[i]
                hyp_uxxxx = utf8_to_uxxxx(hyp_utf8)

                fh_hyp_out.write("%s (%s)\n" % (hyp_uxxxx, metadata['utt-ids'][i]))

                corrected_uttid = metadata['utt-ids'][i]
                corrected_uttid = corrected_uttid[ :corrected_uttid.rfind("_")]
                fh_hyp_out_utf8.write("%s (%s)\n" % (hyp_utf8, corrected_uttid))

            # End of work-loop, mark task as done
            sys.stdout.write(".")
            sys.stdout.flush()


    # When done add newline to stdout
    sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
