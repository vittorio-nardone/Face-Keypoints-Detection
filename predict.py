#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Vittorio Nardone
# DATE CREATED: 05/07/2018
# REVISED DATE:             <=(Date Revised - if any)
# PURPOSE: Use a trained network to detect facial keypoints for an input image
#
# Expected Call with <> indicating expected user input:
#      python predict.py <input> <model>
#
# Example call:
#    python predict.py images/obamas.jpg saved_models/keypoints_model_haar_bn20.pt
#
# Arguments explaination:
# <input> (required)
#     Input image filename
#
# <model> (required)
#     Trained model filename
#
#
# Imports python modules
import argparse
import os.path

import torch
from models import Net
import numpy as np
import cv2
import matplotlib.pyplot as plt


def loadModel(filename):

    net = Net()
    net.load_state_dict(torch.load(filename))
    net.eval()

    print("Network elements:", net)

    return net

def loadFaceDetector(filename=""):

    ## load in a haar cascade classifier for detecting frontal faces
    if filename == "":
        filename = 'detector_architectures/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(filename)

    return face_cascade

def predict(filename, model, detector):

    # load in color image for face detection
    image = cv2.imread(filename)

    # switch red and blue color channels
    # --> by default OpenCV assumes BLUE comes first, not RED as in many images
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #run the detector
    # the output here is an array of detections; the corners of each detection box
    # if necessary, modify these parameters until you successfully identify every face in a given image
    faces = detector.detectMultiScale(image, 1.2, 2)

    # loop over the detected faces from your haar cascade
    for i, (x,y,w,h) in enumerate(faces):

        # Select the region of interest that is the face in the image
        roi = image[y:y+h, x:x+w]

        ## DONE: Convert the face region from RGB to grayscale
        roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        ## DONE: Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]
        roi = roi / 255.0

        ## DONE: Rescale the detected face to be the expected square size for your CNN (224x224, suggested)
        roi = cv2.resize(roi, (224, 224))

        ## DONE: Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)
        roi_t = roi.reshape(1, roi.shape[0], roi.shape[1], 1)
        roi_t = roi_t.transpose((0, 3, 1, 2))
        roi_t = torch.from_numpy(roi_t).type(torch.FloatTensor)

        ## DONE: Make facial keypoint predictions using your loaded, trained network
        predicted_key_pts = model(roi_t)
        predicted_key_pts = predicted_key_pts.view(predicted_key_pts.size()[0], 68, -1).detach().numpy().squeeze()
        predicted_key_pts = predicted_key_pts*50.0+100

        ## DONE: Display each detected face and the corresponding keypoints
        ax = plt.subplot2grid((1, len(faces)), (0, i))
        plt.imshow(roi, cmap='gray')
        plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='b')
        plt.title('Face {}'.format(i))
        plt.axis('off')

    plt.show()

# Main program function defined below
def main():

    #Parse command line arguments
    in_arg = get_input_args()

    #Load face detector
    face_cascade = loadFaceDetector()

    #Load model checkpoint
    model = loadModel(in_arg.model)

    #Prediction
    predict(in_arg.input, model, face_cascade)

# Command line arguments parser
def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object
    """
    # create arguments parser
    parser = argparse.ArgumentParser()

    parser.add_argument("input", nargs=1, type = str,
                    help = "Input image filename")
    parser.add_argument("model", nargs=1, type = str,
                    help = "Trained model filename")
    parser.add_argument('--gpu', action='store_true',
                    help = "If set, GPU is used in prediction (default: False)")

    in_arg = parser.parse_args()

    in_arg.input = in_arg.input[0]
    in_arg.model = in_arg.model[0]

    error_list = []

    # Check input Filename
    if not os.path.isfile(in_arg.input):
        error_list.append("predict.py: error: argument: input: file not found '{}'".format(in_arg.input))

    # Check checkpoint Filename
    if not os.path.isfile(in_arg.model):
        error_list.append("predict.py: error: argument: model: file not found '{}'".format(in_arg.model))

    # Check GPU
    if in_arg.gpu and not mh.gpu_available():
        error_list.append("predict.py: error: argument: --gpu: GPU not available")

    # Print errors
    if len(error_list) > 0:
        parser.print_usage()
        print('\n'.join(error for error in error_list))
        quit()

    # return arguments object
    return in_arg


# Call to main function to run the program
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n** User interruption")
        quit()
