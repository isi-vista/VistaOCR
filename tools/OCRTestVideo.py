#!/bin/env python
import sys, os
import argparse
import datetime

import cv2
import torch
import math
from PIL import Image
from models.cnnlstm import CnnOcrModel
from torch.autograd import Variable
import diagonal_crop
import numpy as np
import utils.decode as dec

def calculateDistance(p1,p2):
     dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
     return dist

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Run OCR with LM for multiple videos in a folder')
    parser.add_argument('-i', '--video-dir', dest='videodir',
                      help="The input video dir")
    parser.add_argument('-v', '--video-list', dest='videolistfile',
                      help="The input video list file")
    parser.add_argument('-b', '--box-dir', dest='boxdir',
                      help="The box dir from text detection")
    parser.add_argument('-m', '--model-path', dest='model_path',
                      help="The ocr model path")
    parser.add_argument('-l', '--lm-path', dest='lm_path',
                      help="The LM model path")
    parser.add_argument('-o', '--output-dir', dest='outputdir',
                      help="The output ocr dir")
    args = parser.parse_args()

    videolistfile = args.videolistfile
    videodir = args.videodir
    boxdir = args.boxdir
    cmodelpath = args.model_path
    outputdir = args.outputdir

    # Load models
    cmodel = CnnOcrModel.FromSavedWeights(args.model_path)

    have_lm = (args.lm_path is not None) and (args.lm_path != "")
    if have_lm:
        lm_units = os.path.join(args.lm_path, 'units.txt')
        lm_words = os.path.join(args.lm_path, 'words.txt')
        lm_wfst = os.path.join(args.lm_path, 'TLG.fst')
        cmodel.init_lm(lm_wfst, lm_words, lm_units, acoustic_weight=0.8)

    line_height = cmodel.input_line_height

    errorlist = []
    imageerrorlist = []
    videolist = []

    lf = open(videolistfile,'r')
    for line in lf.readlines():
        videolist.append(line.strip())
    print(videolist)

    for file in os.listdir(boxdir):
        if not(any(v in file for v in videolist)):
             continue
        videoname = file.split('-')[0]
        print(videoname)

        videomp4 = videoname + '.mp4'
        videofile = os.path.join(videodir,videomp4)
        framenum = file.split('-')[1].split('.')[0]
        boxfilename = file
        print(boxfilename)

        lines = []
        boxfile = os.path.join(boxdir,boxfilename)
        outputname = videoname + '.csv'
        coutputfile = os.path.join(outputdir,outputname)
        try:
             f = open(boxfile, 'r')
        except:
             print("Can't read box file")
             continue
        for line in f.readlines():
             lines.append(line)
        cfile = open(coutputfile,mode = 'a',encoding='utf-8')
        f.close()
        for line in lines:
             x4, y4, x1, y1, x2, y2, x3, y3 = map(int,line.split(','))
             #x1, y1, x2, y2, x3, y3, x4, y4 = map(int,line.split(','))
             cap = cv2.VideoCapture(videofile)
             cap.set(1, int(framenum))
             try:
                  csuccess = True
                  result, frame = cap.read()
                  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                  frame_pil =Image.fromarray(frame)    
                  deltax =x2 - x1
                  deltay =-(y2- y1)
                  angle = math.atan(deltay/deltax)
                  base = (x1,y1)
                  height = calculateDistance((x1,y1),(x4,y4))
                  width = calculateDistance((x1,y1),(x2,y2))
                  crop = diagonal_crop.crop(frame_pil, base, angle, height, width)
                  img = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
                  height,width,channels = img.shape
                  new_width = int(width * (line_height/height))
                  if new_width < line_height:
                      continue
                  img = cv2.resize(img, (new_width,line_height))
                  img_tensor = torch.Tensor(img)
                  img_tensor = img_tensor.permute(2, 0, 1).contiguous()
                  img_np = img_tensor.numpy()
             except:
                  cmodel_hyp = ""
                  csuccess = False
                  imageerrorlist.append(videoname + '-' + framenum)
             if(csuccess == True):
                  try:
                       if have_lm:
                            cmodel_output, cmodel_hyp = dec.decode_single_sample_withlm(cmodel,img_tensor)
                       else:
                            cmodel_output, cmodel_hyp = dec.decode_single_sample(cmodel,img_tensor)
                       csuccess = True
                  except:
                       csuccess = False
                       cmodel_hyp = ""
                  if csuccess == False:
                       errorlist.append(videoname + '-' + framenum + "C:" + str(csuccess))

             s = videoname + ',' + framenum + ',' + str(x4) + ',' + str(y4) + ',' +  str(x1) + ',' + str(y1)+ ',' + str(x2)+ ',' + str(y2) + ',' + str(x3) + ',' + str(y3)+ ',' + str(csuccess) + ',' + cmodel_hyp +  '\n'
             cfile.write(s)

        cfile.close()

    print(errorlist)
    print("EXITED SUCCESSFULLY")


if __name__ == '__main__':
    main()
