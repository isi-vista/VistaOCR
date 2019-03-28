import sys
sys.path.insert(0, '/nas/jschmidt/home/jschmidt/VistaOCR/src')
#sys.path.insert(0, '/nas/jschmidt/home/jschmidt/VistaOCR/src/utils')
import textutils as tu
#from utils import *
#from decode import *
import utils.decode as dec
import torch
#from torch import torch
import cv2
import numpy as np
import imagetransforms
from models.cnnlstm import CnnOcrModel

modelpath = sys.argv[1]
imagepath = sys.argv[2]

model = CnnOcrModel.FromSavedWeights(modelpath)
img = cv2.imread(imagepath,1)
height,width,channels = img.shape
#x = torch.zeros([2, 4])
print(img.shape)
line_height = model.input_line_height
new_width = int(width * (line_height/height))
#img =imagetransforms.Scale(new_h=line_height)
print("Line_height: ",line_height)
print("new width: ",new_width)
img = cv2.resize(img, (new_width,line_height))
print(img.shape)
#img = img.reshape(channels,line_height,new_width)
print(img.shape)
img_tensor = torch.Tensor(img)
img_tensor = img_tensor.permute(2, 0, 1).contiguous()
print(img_tensor.size())
model_output, hyp = dec.decode_single_sample(model,img_tensor)
print(hyp)
