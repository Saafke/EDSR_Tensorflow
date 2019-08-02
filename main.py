import tensorflow as tf 
import data_utils
import run
import os
import cv2
import numpy as np
import pathlib
import argparse
from PIL import Image
import numpy
from tensorflow.python.client import device_lib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #gets rid of avx/fma warning

# TODO: 
# Train models
# Ensemble
# MDSR - doing it in this branch
# decaying learning rate
# make scale dependent data generation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='Train the model', action="store_true")
    parser.add_argument('--test', help='Run tests on the model', action="store_true")
    parser.add_argument('--export', help='Export the model as .pb', action="store_true")
    parser.add_argument('--fromscratch', help='Load previous model for training',action="store_false")
    
    parser.add_argument('--B', type=int, help='Number of resBlocks', default=80)
    parser.add_argument('--F', type=int, help='Number of filters', default=64)
    parser.add_argument('--scale', type=int, help='Scaling factor of the model', default=2)
    parser.add_argument('--batch', type=int, help='Batch size of the training', default=1)
    parser.add_argument('--epochs', type=int, help='Number of epochs during training', default=20)
    parser.add_argument('--lr', type=float, help='Learning_rate', default=0.0001)
    
    parser.add_argument('--image', help='Specify test image', default="./butterfly.png")    
    parser.add_argument('--traindir', help='Path to train images', default="/home/weber/Documents/gsoc/datasets/DIV2K_train_HR")

    args = parser.parse_args()

    # INIT
    scale = args.scale
    meanbgr = [103.1545782, 111.561547, 114.35629928]
    
    # Set checkpoint paths for different scales and models
    ckpt_path = "./CKPT_dir/"
 
    # Set gpu 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Create run instance
    run = run.run(config, ckpt_path, scale, args.batch, args.epochs, args.B, args.F, args.lr, args.fromscratch, meanbgr)

    if args.train:
        run.train(args.traindir)

    if args.test:
        run.test(args.image)
        run.upscale(args.image)

    if args.export:
        run.export()
    
    print("I ran successfully.")