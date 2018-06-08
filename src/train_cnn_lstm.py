import os
import torch
from torch.utils.data import DataLoader
import imagetransforms

import loggy

logger = loggy.setup_custom_logger('root', "train_cnn_lstm.py")

from warpctc_pytorch import CTCLoss
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import datetime
import time
import shutil

from ocr_dataset import OcrDataset
from datautils import GroupedSampler, SortByWidthCollater
from models.cnnlstm import CnnOcrModel
from textutils import *
import argparse


from lr_scheduler import ReduceLROnPlateau

def test_on_val(val_dataloader, model, criterion):
    start_val = time.time()
    cer_running_avg = 0
    wer_running_avg = 0
    loss_running_avg = 0
    n_samples = 0

    display_hyp = True

    # To start, put model in eval mode
    model.eval()

    logger.info("About to comptue %d val batches" % len(val_dataloader))

    # No need for backprop during validation test
    with torch.no_grad():
        for input_tensor, target, input_widths, target_widths, metadata in val_dataloader:
            input_tensor = input_tensor.cuda(async=True)

            model_output, model_output_actual_lengths = model(input_tensor, input_widths)
            loss = criterion(model_output, target, model_output_actual_lengths, target_widths)

            hyp_transcriptions = model.decode_without_lm(model_output, model_output_actual_lengths, uxxxx=True)

            batch_size = input_tensor.size(0)
            curr_loss = loss.data[0] / batch_size
            n_samples += 1
            loss_running_avg += (curr_loss - loss_running_avg) / n_samples

            cur_target_offset = 0
            batch_cer = 0
            batch_wer = 0
            target_np = target.data.numpy()
            ref_transcriptions = []
            for i in range(len(hyp_transcriptions)):
                ref_transcription = form_target_transcription(
                    target_np[cur_target_offset:(cur_target_offset + target_widths.data[i])],
                    model.alphabet
                )
                ref_transcriptions.append(uxxxx_to_utf8(ref_transcription))
                cur_target_offset += target_widths.data[i]
                cer, wer = compute_cer_wer(hyp_transcriptions[i], ref_transcription)

                batch_cer += cer
                batch_wer += wer

            cer_running_avg += (batch_cer / batch_size - cer_running_avg) / n_samples
            wer_running_avg += (batch_wer / batch_size - wer_running_avg) / n_samples

            # For now let's display one set of transcriptions every test, just to see improvements
            if display_hyp:
                logger.info("--------------------")
                logger.info("Sample hypothesis / reference transcripts")
                logger.info("Error rate for this batch is:\tNo LM CER: %f\tWER:%f" % (
                batch_cer / batch_size, batch_wer / batch_size))

                hyp_transcriptions = model.decode_without_lm(model_output, model_output_actual_lengths, uxxxx=False)
                for i in range(len(hyp_transcriptions)):
                    logger.info("\tHyp[%d]: %s" % (i, hyp_transcriptions[i]))
                    logger.info("\tRef[%d]: %s" % (i, ref_transcriptions[i]))
                    logger.info("")
                logger.info("--------------------")
                display_hyp = False

    # Finally, put model back in train mode
    model.train()
    end_val = time.time()
    logger.info("Total val time: %s" % (end_val - start_val))
    return loss_running_avg, cer_running_avg, wer_running_avg


def train(batch, model, criterion, optimizer):
    input_tensor, target, input_widths, target_widths, metadata = batch
    input_tensor = input_tensor.cuda(async=True)

    optimizer.zero_grad()
    model_output, model_output_actual_lengths = model(input_tensor, input_widths)

    loss = criterion(model_output, target, model_output_actual_lengths, target_widths)
    loss.backward()
    
    # RNN Backprop can have exploding gradients (even with LSTM), so make sure
    # we clamp the abs magnitude of individual gradient entries
    for param in model.parameters():
        if not param.grad is None:
            param.grad.data.clamp_(min=-5, max=5)


    # Okay, now we're ready to update parameters!
    optimizer.step()
    return loss.data[0]


def get_args():
    parser = argparse.ArgumentParser(description="OCR Training Script")
    parser.add_argument("--batch-size", type=int, default=64, help="SGD mini-batch size")
    parser.add_argument("--num_in_channels", type=int, default=1, help="Number of input channels for image (1 for grayscale or 3 for color)")
    parser.add_argument("--line-height", type=int, default=30, help="Input image line height")
    parser.add_argument("--rds-line-height", type=int, default=30, help="Target line height after rapid-downsample layer")
    parser.add_argument("--datadir", type=str, required=True, help="specify the location to data.")
    parser.add_argument("--snapshot-prefix", type=str, required=True,
                        help="Output directory and basename prefix for output model snapshots")
    parser.add_argument("--load-from-snapshot", type=str,
                        help="Path to snapshot from which we should initialize model weights")
    parser.add_argument("--num-lstm-layers", type=int, required=True, help="Number of LSTM layers in model")
    parser.add_argument("--num-lstm-units", type=int, required=True,
                        help="Number of LSTM hidden units in each LSTM layer (single number, or comma seperated list)")
    parser.add_argument("--lstm-input-dim", type=int, required=True, help="Input dimension for LSTM")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--nepochs", type=int, default=250, help="Maximum number of epochs to train")
    parser.add_argument("--snapshot-num-iterations", type=int, default=2000, help="Every N iterations snapshot model")
    parser.add_argument("--patience", type=int, default=10, help="Patience parameter for ReduceLROnPlateau.")
    parser.add_argument("--min-lr", type=float, default=1e-7, help="Minimum learning rate for ReduceLROnPlateau")
    parser.add_argument("--rtl", default=False, action='store_true', help="Set if language is right-to-left")
    parser.add_argument("--synth_input", default=False, action='store_true', help="Specifies if input data is synthetic; if so we apply extra data augmentation")
    return parser.parse_args()


def main():
    logger.info("Starting training\n\n")
    sys.stdout.flush()
    args = get_args()
    snapshot_path = args.snapshot_prefix + "-cur_snapshot.pth"
    best_model_path = args.snapshot_prefix + "-best_model.pth"


    line_img_transforms = []

    # Data augmentations (during training only)
    if args.synth_input:
        line_img_transforms.append( imagetransforms.DegradeDownsample(ds_factor=0.2) )


    # Make sure to do resize after degrade step above
    line_img_transforms.append(imagetransforms.Scale(new_h=args.line_height))

    # Only do for grayscale
    if args.num_in_channels == 1:
        line_img_transforms.append(imagetransforms.InvertBlackWhite())

    # For right-to-left languages
    if args.rtl:
        print("Right to Left")
        line_img_transforms.append(imagetransforms.HorizontalFlip())


    line_img_transforms.append(imagetransforms.ToTensor())

    line_img_transforms = imagetransforms.Compose(line_img_transforms)


    # Setup cudnn benchmarks for faster code
    torch.backends.cudnn.benchmark = False

    train_dataset = OcrDataset(args.datadir, "train", line_img_transforms)
    validation_dataset = OcrDataset(args.datadir, "validation", line_img_transforms)

    train_dataloader = DataLoader(train_dataset, args.batch_size, num_workers=4, sampler=GroupedSampler(train_dataset, rand=True),
                                  collate_fn=SortByWidthCollater, pin_memory=True, drop_last=True)

    validation_dataloader = DataLoader(validation_dataset, args.batch_size, num_workers=0,sampler=GroupedSampler(validation_dataset, rand=False),
                                       collate_fn=SortByWidthCollater, pin_memory=False, drop_last=False)


    n_epochs = args.nepochs
    lr_alpha = args.lr
    snapshot_every_n_iterations = args.snapshot_num_iterations

    if args.load_from_snapshot is not None:
        model = CnnOcrModel.FromSavedWeights(args.load_from_snapshot)
    else:
        model = CnnOcrModel(
            num_in_channels=args.num_in_channels,
            input_line_height=args.line_height,
            rds_line_height=args.rds_line_height,
            lstm_input_dim=args.lstm_input_dim,
            num_lstm_layers=args.num_lstm_layers,
            num_lstm_hidden_units=args.num_lstm_units,
            p_lstm_dropout=0.5,
            alphabet=train_dataset.alphabet,
            multigpu=True)

    # Set training mode on all sub-modules
    model.train()

    ctc_loss = CTCLoss().cuda()

    iteration = 0
    best_val_wer = float('inf')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_alpha)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=args.patience, min_lr=args.min_lr)
    wer_array = []
    cer_array = []
    loss_array = []
    lr_points = []
    iteration_points = []

    epoch_size = len(train_dataloader)

    for epoch in range(1, n_epochs + 1):
        epoch_start = datetime.datetime.now()

        # First modify main OCR model
        for batch in train_dataloader:
            sys.stdout.flush()
            iteration += 1
            iteration_start = datetime.datetime.now()


            loss = train(batch, model, ctc_loss, optimizer)

            elapsed_time = datetime.datetime.now() - iteration_start
            loss = loss / args.batch_size

            loss_array.append(loss)

            logger.info("Iteration: %d (%d/%d in epoch %d)\tLoss: %f\tElapsed Time: %s" % (
            iteration, iteration % epoch_size, epoch_size, epoch, loss, pretty_print_timespan(elapsed_time)))

            # Do something with loss, running average, plot to some backend server, etc

            if iteration % snapshot_every_n_iterations == 0:
                logger.info("Testing on validation set")
                val_loss, val_cer, val_wer = test_on_val(validation_dataloader, model, ctc_loss)
                # Reduce learning rate on plateau
                early_exit = False
                lowered_lr = False
                if scheduler.step(val_wer):
                    lowered_lr = True
                    lr_points.append(iteration / snapshot_every_n_iterations)
                    if scheduler.finished:
                        early_exit = True

                    # for bookeeping only
                    lr_alpha = max(lr_alpha * scheduler.factor, scheduler.min_lr)

                logger.info("Val Loss: %f\tNo LM Val CER: %f\tNo LM Val WER: %f" % (val_loss, val_cer, val_wer))

                torch.save({'iteration': iteration,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_hyper_params': model.get_hyper_params(),
                            'rtl': args.rtl,
                            'cur_lr': lr_alpha,
                            'val_loss': val_loss,
                            'val_cer': val_cer,
                            'val_wer': val_wer,
                            'line_height': args.line_height
                        },
                           snapshot_path)

                # plotting lr_change on wer, cer and loss.
                wer_array.append(val_wer)
                cer_array.append(val_cer)
                iteration_points.append(iteration / snapshot_every_n_iterations)

                if val_wer < best_val_wer:
                    logger.info("Best model so far, copying snapshot to best model file")
                    best_val_wer = val_wer
                    shutil.copyfile(snapshot_path, best_model_path)

                logger.info("Running WER: %s" % str(wer_array))
                logger.info("Done with validation, moving on.")

                if early_exit:
                    logger.info("Early exit")
                    sys.exit(0)

                if lowered_lr:
                    logger.info("Switching to best model parameters before continuing with lower LR")
                    weights = torch.load(best_model_path)
                    model.load_state_dict(weights['state_dict'])


        elapsed_time = datetime.datetime.now() - epoch_start
        logger.info("\n------------------")
        logger.info("Done with epoch, elapsed time = %s" % pretty_print_timespan(elapsed_time))
        logger.info("------------------\n")


    #writer.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
