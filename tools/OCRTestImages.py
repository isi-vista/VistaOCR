#!/bin/env python
import h5py as h5
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from PIL import Image
import sys
import diagonal_crop
import json
import lmdb
import os
import textutils as tu
import imagetransforms
from models.cnnlstm import CnnOcrModel
from models.cnn_script_id_mean import CnnScriptIdModelMean
from models.cnn_script_id_max import CnnScriptIdModelMax
from models.cnn_script_id_lstm import CnnScriptIdLstmModel
import utils.decode as dec
import torch
import random

def calculateDistance(p1,p2):
     dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
     return dist

def main(argv=None):
    if argv is None:
          argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Run OCR for multiple images in a folder')
    parser.add_argument('-i', '--image-dir', dest='imagedir',
                      help="The input image dir")
    parser.add_argument('-b', '--box-dir', dest='boxdir',
                      help="The box dir from text detection")
    parser.add_argument('-m', '--model-path', dest='cmodelpath',
                      help="The ocr model path")
    parser.add_argument('-l', '--lm-path', dest='lm_path',
                      help="The LM model path")
    parser.add_argument('-o', '--output-dir', dest='outputdir',
                      help="The output ocr dir")
    args = parser.parse_args()

    boxdir = args.boxdir
    imagedir = args.imagedir
    cmodelpath = args.cmodelpath
    outputdir = args.outputdir

    cmodel = CnnOcrModel.FromSavedWeights(cmodelpath)
    line_height = cmodel.input_line_height

    # Load LM
    have_lm = (args.lm_path is not None) and (args.lm_path != "")
    if have_lm:
        lm_units = os.path.join(args.lm_path, 'units.txt')
        lm_words = os.path.join(args.lm_path, 'words.txt')
        lm_wfst = os.path.join(args.lm_path, 'TLG.fst')
        cmodel.init_lm(lm_wfst, lm_words, lm_units, acoustic_weight=0.8)

    errorlist = []
    for file in os.listdir(imagedir):
        imagename = file.split('.')[0]
        print(file)

        framenum = imagename.split('_')[-1]
        imagefile = os.path.join(imagedir,file)
        boxfilename = imagename + ".txt"
        print(boxfilename)

        boxfile = os.path.join(boxdir,boxfilename)
        outputname = imagename + '.csv'
        coutputfile = os.path.join(outputdir,outputname)
        f = open(boxfile, 'r')
        lines = []
        for line in f.readlines():
             lines.append(line)
             f.close()

        cfile = open(coutputfile,'a')

        for line in lines:
             x4, y4, x1, y1, x2, y2, x3, y3 = map(int,line.split(','))
             #x1, y1, x2, y2, x3, y3, x4, y4 = map(int,line.split(','))
             frame = cv2.imread(imagefile,1)
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
             try:
                  if have_lm:
                        cmodel_output, cmodel_hyp = dec.decode_single_sample_withlm(cmodel,img_tensor)
                  else:
                        cmodel_output, cmodel_hyp = dec.decode_single_sample(cmodel,img_tensor)
                  csuccess = True
             except:
                  csuccess = False
                  cmodel_hyp = ""

             s = imagename + ',' + framenum + ',' + str(x4) + ',' + str(y4) + ',' +  str(x1) + ',' + str(y1)+ ',' + str(x2)+ ',' + str(y2) + ',' + str(x3) + ',' + str(y3)+ ',' + str(csuccess) + ',' + cmodel_hyp +  '\n'
             cfile.write(s)

        cfile.close()

    print(errorlist)

if __name__ == '__main__':
    main()
