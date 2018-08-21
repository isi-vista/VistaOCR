import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import sys
import argparse
import time
from PIL import Image
import cv2
import csv
import os


class ImageAug(object):
    
    
    def __init__(self):
        # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
        # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
        sometimes = lambda aug: iaa.Sometimes(.90, aug)
      
        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        seq = iaa.Sequential(
            [
                sometimes(iaa.CropAndPad(
                    percent=(-0.03, 0.03),
                    pad_mode=["constant", "edge"],
                    pad_cval=(0, 255)
                )),
                sometimes(iaa.Affine(
                    #scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                    #translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
                    rotate=(-3, 3), # rotate by -45 to +45 degrees
                    shear=(-3, 3), # shear by -16 to +16 degrees
                    #order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    #cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    #mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                iaa.SomeOf((0, 3),
                    [
                        #iaa.OneOf([
                        #    iaa.GaussianBlur((0, 0.8)), # blur images with a sigma between 0 and 3.0
                        #]),
                        iaa.GaussianBlur((0, 0.75)), # blur images with a sigma between 0 and 3.0
                        iaa.Sharpen(alpha=(0, 0.5), lightness=(0.75, 1.5)), # sharpen images
                        iaa.Emboss(alpha=(0, 0.5), strength=(0, 1.5)), # emboss images
                        #iaa.SimplexNoiseAlpha(iaa.OneOf([
                        #    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                        #    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                        #])),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                        iaa.Dropout((0.01, 0.05), per_channel=0.5), # randomly remove up to 10% of the pixels
                        #iaa.OneOf([
                        #    iaa.Dropout((0.01, 0.05), per_channel=0.5), # randomly remove up to 10% of the pixels
                        #    iaa.CoarseDropout((0.01, 0.05), size_percent=(0.02, 0.04), per_channel=0.2),
                        #]),
                        iaa.Invert(0.05, per_channel=True), # invert color channels
                        iaa.Add((-5, 5), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                        iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                        #iaa.OneOf([
                        #    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                        #    iaa.FrequencyNoiseAlpha(
                        #        exponent=(-4, 0),
                        #        first=iaa.Multiply((0.5, 1.5), per_channel=True),
                        #        second=iaa.ContrastNormalization((0.5, 2.0))
                        #    )
                        #]),
                        iaa.ContrastNormalization((0.5, 1.0), per_channel=0.5), # improve or worsen the contrast
                        iaa.Grayscale(alpha=(0.0, 1.0)),
                    ],
                    random_order=True
                )
            ],
            random_order=True
        )
        self.seq = seq


    def __call__(self, img):
        images_aug = self.seq.augment_image(img)
        return images_aug
    

