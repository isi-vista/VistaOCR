import os
import torch
from torch.utils.data import DataLoader
import imagetransforms
from imgaug import augmenters as iaa
import daves_augment

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
from ocr_dataset_union import OcrDatasetUnion
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

def test_on_val_writeout(val_dataloader, model, out_hyp_path):
    start_val = time.time()

    # To start, put model in eval mode
    model.eval()

    logger.info("About to comptue %d val batches" % len(val_dataloader))

    # No need for backprop during validation test
    with torch.no_grad(), open(out_hyp_path, 'w') as fh_out:
        for input_tensor, target, input_widths, target_widths, metadata in val_dataloader:
            input_tensor = input_tensor.cuda(async=True)

            model_output, model_output_actual_lengths = model(input_tensor, input_widths)
            hyp_transcriptions = model.decode_without_lm(model_output, model_output_actual_lengths, uxxxx=False)

            for i in range(len(hyp_transcriptions)):
                hyp_utf8 = utf8_visual_to_logical(hyp_transcriptions[i])
                uttid = metadata['utt-ids'][i]
                fh_out.write("%s (%s)\n" % (hyp_utf8, uttid))


    # Finally, put model back in train mode
    model.train()
    end_val = time.time()
    logger.info("Total decode + write time is: %s" % (end_val - start_val))
    return


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
    parser.add_argument("--datadir", type=str, action='append', required=True, help="specify the location to data.")
    parser.add_argument("--test-datadir", type=str, default=None, help="optionally produce hyps on test set every validation pass; this is data dir")
    parser.add_argument("--test-outdir", type=str, default=None, help="optionally produce hyps on test set every validation pass; this is output dir to place hyps")
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
    parser.add_argument("--weight-decay", type=float, default=0, help="Weight decay l2 regularization term")
    parser.add_argument("--rtl", default=False, action='store_true', help="Set if language is right-to-left")
    parser.add_argument("--cvtGray", default=False, action='store_true', help="Set if need to convert color images to grayscale") 
    parser.add_argument("--max-val-size", type=int, default=-1, help="If validation set is large, limit it to a smaller size for faster validation runs")
    parser.add_argument("--synth_input", default=False, action='store_true', help="Specifies if input data is synthetic; if so we apply extra data augmentation")
    parser.add_argument("--daves_augment", default=False, action='store_true', help="Specifies if we should apply daves seq of data augmentations")
    parser.add_argument("--tight_crop", default=False, action='store_true', help="Specifies if synthetic data is tight-crop or not; if so we change padding")
    parser.add_argument("--stripe", default=False, action='store_true', help="Specifies if we add random white strips to image as data augmentation")
    return parser.parse_args()


def main():
    logger.info("Starting training\n\n")
    sys.stdout.flush()
    args = get_args()
    snapshot_path = args.snapshot_prefix + "-cur_snapshot.pth"
    best_model_path = args.snapshot_prefix + "-best_model.pth"

    line_img_transforms = []

    #if args.num_in_channels == 3:
    #    line_img_transforms.append(imagetransforms.ConvertColor())

    # Always convert color for the augmentations to work (for now)
    # Then alter convert back to grayscale if needed
    line_img_transforms.append(imagetransforms.ConvertColor())


    # Data augmentations (during training only)
    if args.daves_augment:
        line_img_transforms.append(daves_augment.ImageAug())

    if args.synth_input:

        # Randomly rotate image from -2 degrees to +2 degrees
        line_img_transforms.append(imagetransforms.Randomize(0.3, imagetransforms.RotateRandom(-2,2)))

        # Choose one of methods to blur/pixel-ify image  (or don't and choose identity)
        line_img_transforms.append(
            imagetransforms.PickOne([
                imagetransforms.TessBlockConv(kernel_val=1, bias_val=1),
                imagetransforms.TessBlockConv(rand=True),
                imagetransforms.Identity(),
            ])
        )

        aug_cn = iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)
        line_img_transforms.append( imagetransforms.Randomize(0.5, lambda x: aug_cn.augment_image(x)) )

        # With some probability, choose one of:
        #   Grayscale:  convert to grayscale and add back into color-image with random alpha
        #   Emboss:  Emboss image with random strength
        #   Invert:  Invert colors of image per-channel
        aug_gray = iaa.Grayscale(alpha=(0.0, 1.0))
        aug_emboss = iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))
        aug_invert = iaa.Invert(1, per_channel=True)
        aug_invert2 = iaa.Invert(0.1, per_channel=False)
        line_img_transforms.append( imagetransforms.Randomize(0.3,
                                                              imagetransforms.PickOne([
                                                                  lambda x: aug_gray.augment_image(x),
                                                                  lambda x: aug_emboss.augment_image(x),
                                                                  lambda x: aug_invert.augment_image(x),
                                                                  lambda x: aug_invert2.augment_image(x)
                                                                  ])) )


        # Randomly try to crop close to top/bottom and left/right of lines
        # For now we are just guessing (up to 5% of ends and up to 10% of tops/bottoms chopped off)

        if args.tight_crop:
            # To make sure padding is reasonably consistent, we first rsize image to target line height
            # Then add padding to this version of image
            # Below it will get resized again to target line height
            line_img_transforms.append( imagetransforms.Randomize(0.9, imagetransforms.Compose([
                imagetransforms.Scale(new_h=args.line_height),
                imagetransforms.PadRandom(pxl_max_horizontal=30, pxl_max_vertical=10)
                ])))

        else:
            line_img_transforms.append( imagetransforms.Randomize(0.2, imagetransforms.CropHorizontal(.05)) )
            line_img_transforms.append( imagetransforms.Randomize(0.2, imagetransforms.CropVertical(.1)) )

        #line_img_transforms.append(imagetransforms.Randomize(0.2,
        #                                                     imagetransforms.PickOne([imagetransforms.MorphErode(3), imagetransforms.MorphDilate(3)])
        #                                                     ))


    # Make sure to do resize after degrade step above
    line_img_transforms.append(imagetransforms.Scale(new_h=args.line_height))


    if args.cvtGray:
        line_img_transforms.append(imagetransforms.ConvertGray())

    # Only do for grayscale
    if args.num_in_channels == 1:
        line_img_transforms.append(imagetransforms.InvertBlackWhite())

    if args.stripe:
        line_img_transforms.append(imagetransforms.Randomize(0.3, imagetransforms.AddRandomStripe(val=0, strip_width_from=1, strip_width_to=4)))


    line_img_transforms.append(imagetransforms.ToTensor())

    line_img_transforms = imagetransforms.Compose(line_img_transforms)


    # Setup cudnn benchmarks for faster code
    torch.backends.cudnn.benchmark = False

    if len(args.datadir) == 1:
        train_dataset = OcrDataset(args.datadir[0], "train", line_img_transforms)
        validation_dataset = OcrDataset(args.datadir[0], "validation", line_img_transforms)
    else:
        train_dataset = OcrDatasetUnion(args.datadir, "train", line_img_transforms)
        validation_dataset = OcrDatasetUnion(args.datadir, "validation", line_img_transforms)


    if args.test_datadir is not None:
        if args.test_outdir is None:
            print("Error, must specify both --test-datadir and --test-outdir together")
            sys.exit(1)

        if not os.path.exists(args.test_outdir):
            os.makedirs(args.test_outdir)

        line_img_transforms_test = imagetransforms.Compose([imagetransforms.Scale(new_h=args.line_height), imagetransforms.ToTensor()])
        test_dataset = OcrDataset(args.test_datadir, "test", line_img_transforms_test)


    n_epochs = args.nepochs
    lr_alpha = args.lr
    snapshot_every_n_iterations = args.snapshot_num_iterations

    if args.load_from_snapshot is not None:
        model = CnnOcrModel.FromSavedWeights(args.load_from_snapshot)
        print("Overriding automatically learned alphabet with pre-saved model alphabet")
        if len(args.datadir) == 1:
            train_dataset.alphabet = model.alphabet
            validation_dataset.alphabet = model.alphabet
        else:
            train_dataset.alphabet = model.alphabet
            validation_dataset.alphabet = model.alphabet
            for ds in train_dataset.datasets:
                ds.alphabet = model.alphabet
            for ds in validation_dataset.datasets:
                ds.alphabet = model.alphabet

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


    # Setting dataloader after we have a chnae to (maybe!) over-ride the dataset alphabet from a pre-trained model
    train_dataloader = DataLoader(train_dataset, args.batch_size, num_workers=4, sampler=GroupedSampler(train_dataset, rand=True),
                                  collate_fn=SortByWidthCollater, pin_memory=True, drop_last=True)

    if args.max_val_size > 0:
        validation_dataloader = DataLoader(validation_dataset, args.batch_size, num_workers=0,sampler=GroupedSampler(validation_dataset, max_items=args.max_val_size, fixed_rand=True),
                                           collate_fn=SortByWidthCollater, pin_memory=False, drop_last=False)
    else:
        validation_dataloader = DataLoader(validation_dataset, args.batch_size, num_workers=0,sampler=GroupedSampler(validation_dataset, rand=False),
                                           collate_fn=SortByWidthCollater, pin_memory=False, drop_last=False)



    if args.test_datadir is not None:
        test_dataloader = DataLoader(test_dataset, args.batch_size, num_workers=0,sampler=GroupedSampler(test_dataset, rand=False),
                                     collate_fn=SortByWidthCollater, pin_memory=False, drop_last=False)



    # Set training mode on all sub-modules
    model.train()

    ctc_loss = CTCLoss().cuda()

    iteration = 0
    best_val_wer = float('inf')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_alpha, weight_decay=args.weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=args.patience, min_lr=args.min_lr)
    wer_array = []
    cer_array = []
    loss_array = []
    lr_points = []
    iteration_points = []

    epoch_size = len(train_dataloader)

    do_test_write = False
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

            # Only turn on test-on-testset when cer is starting to get non-random
            if iteration % snapshot_every_n_iterations == 0:
                logger.info("Testing on validation set")
                val_loss, val_cer, val_wer = test_on_val(validation_dataloader, model, ctc_loss)

                if val_cer < 0.5:
                    do_test_write = True

                if args.test_datadir is not None and (iteration % snapshot_every_n_iterations == 0) and do_test_write:
                    out_hyp_outdomain_file = os.path.join(args.test_outdir, "hyp-%07d.outdomain.utf8" % iteration)
                    out_hyp_indomain_file = os.path.join(args.test_outdir, "hyp-%07d.indomain.utf8" % iteration)
                    out_meta_file = os.path.join(args.test_outdir, "hyp-%07d.meta" % iteration)
                    test_on_val_writeout(test_dataloader, model, out_hyp_outdomain_file)
                    test_on_val_writeout(validation_dataloader, model, out_hyp_indomain_file)
                    with open(out_meta_file, 'w') as fh_out:
                        fh_out.write("%d,%f,%f,%f\n" % (iteration, val_cer, val_wer, val_loss))


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
