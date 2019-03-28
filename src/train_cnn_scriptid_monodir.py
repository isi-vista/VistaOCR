import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import imagetransforms

import loggy

logger = loggy.setup_custom_logger('root', "train_cnn_scriptid_lstm.py")

from warpctc_pytorch import CTCLoss
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import datetime
import time
import shutil

from script_id_dataset import ScriptIdDataset
from datautils import GroupedSampler, SortByWidthCollater_scriptid_lstm
from models.cnn_script_id_lstm_monodir import CnnScriptIdLstmModel
from textutils import *
import argparse


from lr_scheduler import ReduceLROnPlateau

def test_on_val(val_dataloader, model, criterion):
    start_val = time.time()
    loss_running_avg = 0
    n_correct = 0
    n_samples = 0
    n_russian_hyp = 0
    display_hyp = True

    # To start, put model in eval mode
    model.eval()

    logger.info("About to comptue %d val batches" % len(val_dataloader))
    for input_tensor, target, input_widths, metadata in val_dataloader:
        # In validation set, not doing backprop, so set volatile to True to reduce memory footprint
        input_tensor = Variable(input_tensor.cuda(async=True))
        target = Variable(target).cuda()
        input_widths = Variable(input_widths)
        model_output,actual_widths = model(input_tensor, input_widths)
        rect_output=model_output.view(-1,2)
        masklist = []
        maxlength = model_output.size(0)
        actual_samples = 0
        for i in range(0,maxlength):
            for length in actual_widths:
                if i >= length:
                    masklist.append(1)
                else:
                    actual_samples += 1
                    masklist.append(0)
        target_mask = torch.ByteTensor(masklist).cuda()
        rep_target = target.repeat(maxlength)
        final_target = rep_target.masked_fill(Variable(target_mask),2)
        loss = criterion(rect_output, final_target)
        #batch_size = input_tensor.size(0)
        batch_size = input_tensor.size(0)
        n_samples += actual_samples
        curr_loss = loss.data[0]/actual_samples
        loss_val = float(curr_loss - loss_running_avg) / n_samples
        loss_running_avg += float(curr_loss - loss_running_avg) / n_samples
        prob_output = torch.nn.functional.softmax(rect_output,1)
        #for idx in range(0,prob_output.size(1)):
         #   if(prob_output.data[actual_widths[idx]-1,idx, 0] > 0.5):
         #       label = 0
         #       n_russian_hyp += 1
         #   else:
         #       label = 1
         #   if label == target.data[idx]:
         #       n_correct += 1

        for idx in range(0,prob_output.size(0)):
            if(prob_output.data[idx, 0] > 0.5):
                label = 0
                if(not final_target.data[idx] == 2):
                    n_russian_hyp += 1
            else:
                label = 1
            if label == final_target.data[idx]:
                n_correct += 1
    print(prob_output.data[0,0], prob_output.data[0,1])
    print(prob_output.data[1,0], prob_output.data[1,1])
    print(prob_output.data[2,0], prob_output.data[2,1])
    print(prob_output.data[3,0], prob_output.data[3,1])
    # Finally, put model back in train mode
    model.train()
    end_val = time.time()
    accuracy = n_correct / n_samples
    russian_percent = float(n_russian_hyp)/n_samples
    logger.info("Total val time: %s. Accuracy = %f" % ((end_val - start_val), accuracy))
    logger.info("Total val time: %s. Loss_val = %f Curr_loss = %f" % ((end_val - start_val), loss_val,curr_loss))
    logger.info("Total val time: %s. Percent Russian = %f N_correct %f N_samples %f" % ((end_val - start_val), russian_percent, n_correct, n_samples))
  
  
    return loss_running_avg, accuracy


def train(batch, model, criterion, optimizer):
    input_tensor, target,input_widths,metadata  = batch
    input_tensor = Variable(input_tensor.cuda(async=True))
    input_widths = Variable(input_widths)
    target = Variable(target).cuda()
    optimizer.zero_grad()
    model_output,actual_widths= model(input_tensor,input_widths)
    torch.set_printoptions(threshold=5000)
    #print("Model output: ",model_output.size())
    #print("Target: ", target.size())
    #print("Target: ", target)
    rect_output=model_output.view(-1,2)
    #print("Actual widths: ",actual_widths)
    #print(len(actual_widths))
    
    masklist = []
    maxlength = model_output.size(0)
    #print("maxlength:",maxlength)
#    for length in actual_widths:
#        for i in range(0,maxlength):

    for i in range(0,maxlength):
        for length in actual_widths:
            if i >= length:
                masklist.append(1)
            else:
                masklist.append(0)
    target_mask = torch.ByteTensor(masklist).cuda()
    rep_target = target.repeat(maxlength)
    #print(rep_target)
    final_target = rep_target.masked_fill(Variable(target_mask),2)
    #print(final_target)
    loss = criterion(rect_output, final_target)
    loss.backward()
    optimizer.step()
    return loss.data[0]


def get_args():
    parser = argparse.ArgumentParser(description="OCR Training Script")
    parser.add_argument("--batch-size", type=int, default=64, help="SGD mini-batch size")
    parser.add_argument("--line-height", type=int, default=30, help="Input image line height")
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
    return parser.parse_args()


def main():
    logger.info("Starting training\n\n")
    sys.stdout.flush()
    args = get_args()
    snapshot_path = args.snapshot_prefix + "-cur_snapshot.pth"
    best_model_path = args.snapshot_prefix + "-best_model.pth"


    line_img_transforms = imagetransforms.Compose([
        imagetransforms.Scale(new_h=args.line_height),
        imagetransforms.InvertBlackWhite(),
        imagetransforms.ToTensor(),
    ])


    # Setup cudnn benchmarks for faster code
    torch.backends.cudnn.benchmark = False

    train_dataset = ScriptIdDataset(args.datadir, "train", line_img_transforms)
    validation_dataset = ScriptIdDataset(args.datadir, "validation", line_img_transforms)

    train_dataloader = DataLoader(train_dataset, args.batch_size, num_workers=4, sampler=GroupedSampler(train_dataset, rand=True),collate_fn=SortByWidthCollater_scriptid_lstm,
                                  pin_memory=True, drop_last=True)

    validation_dataloader = DataLoader(validation_dataset, args.batch_size, num_workers=0,sampler=GroupedSampler(validation_dataset, rand=False), collate_fn=SortByWidthCollater_scriptid_lstm,
                                       pin_memory=False, drop_last=False)


    n_epochs = args.nepochs
    lr_alpha = args.lr
    snapshot_every_n_iterations = args.snapshot_num_iterations

    if args.load_from_snapshot is not None:
        model = CnnScriptIdLstmModel.FromSavedWeights(args.load_from_snapshot)
    else:
        model = CnnScriptIdLstmModel(
            input_line_height=args.line_height,
            lstm_input_dim=args.lstm_input_dim,
            num_lstm_layers=args.num_lstm_layers,
            num_lstm_hidden_units=args.num_lstm_units,
            p_lstm_dropout=0.5,
            num_in_channels=3,
            multigpu=True)

    # Set training mode on all sub-modules
    model.train()

    criterion = nn.CrossEntropyLoss(ignore_index=2)

    iteration = 0
    best_val_loss = float('inf')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_alpha)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=args.patience, min_lr=args.min_lr)
    accuracy_array = []
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


            loss = train(batch, model, criterion, optimizer)

            elapsed_time = datetime.datetime.now() - iteration_start
            loss = loss / args.batch_size

            loss_array.append(loss)

            logger.info("Iteration: %d (%d/%d in epoch %d)\tLoss: %f\tElapsed Time: %s" % (
            iteration, iteration % epoch_size, epoch_size, epoch, loss, pretty_print_timespan(elapsed_time)))

            # Do something with loss, running average, plot to some backend server, etc

            if iteration % snapshot_every_n_iterations == 0:
                logger.info("Testing on validation set")
                val_loss, val_accuracy = test_on_val(validation_dataloader, model, criterion)
                accuracy_array.append(val_accuracy)
                # Reduce learning rate on plateau
                early_exit = False
                lowered_lr = False
                if scheduler.step(val_loss):
                    lowered_lr = True
                    lr_points.append(iteration / snapshot_every_n_iterations)
                    if scheduler.finished:
                        early_exit = True

                    # for bookeeping only
                    lr_alpha = max(lr_alpha * scheduler.factor, scheduler.min_lr)

                logger.info("Val Loss: %f\tVal Accuracy: %f" % (val_loss, val_accuracy))

                torch.save({'iteration': iteration,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_hyper_params': model.get_hyper_params(),
                            'cur_lr': lr_alpha,
                            'val_loss': val_loss,
                            'val_accuracy': val_accuracy,
                            'line_height': args.line_height
                        },
                           snapshot_path)

                iteration_points.append(iteration / snapshot_every_n_iterations)

                if val_loss < best_val_loss:
                    logger.info("Best model so far, copying snapshot to best model file")
                    best_val_loss = val_loss
                    shutil.copyfile(snapshot_path, best_model_path)

                logger.info("Running Accuracy over time: %s" % str(accuracy_array))
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
