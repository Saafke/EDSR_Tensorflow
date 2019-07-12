import tensorflow as tf 
import edsr
import utils
import run_utils
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
# Subtract mean rgb - 0.57880239, 0.45851252, 0.33622327
# Divide by standev - 0.21396621, 0.1942685 , 0.20073332
# Download div2k
# [DONE] L1-loss
# Ensemble
# [DONE] RGB data loading
# Create validation set

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='Train the model', action="store_true")
    parser.add_argument('--test', help='Run tests on the model', action="store_true")
    parser.add_argument('--export', help='Export the model as .pb', action="store_true")
    parser.add_argument('--fromscratch', help='Load previous model for training',action="store_false")
    parser.add_argument('--finetune', help='Finetune model on General100 dataset',action="store_true")
    parser.add_argument('--B', type=int, help='Number of resBlocks', default=32)
    parser.add_argument('--F', type=int, help='Number of filters', default=256)
    parser.add_argument('--scale', type=int, help='Scaling factor of the model', default=2)
    parser.add_argument('--batch', type=int, help='Batch size of the training', default=1)
    parser.add_argument('--epochs', type=int, help='Number of epochs during training', default=20)
    parser.add_argument('--image', help='Specify test image', default="./butterfly.png")
    parser.add_argument('--lr', type=float, help='Learning_rate', default=0.001)
    parser.add_argument('--traindir', help='Path to train images')
    parser.add_argument('--general100_dir', help='Path to General100 dataset')

    args = parser.parse_args()

    # INIT
    B = args.B
    F = args.F
    scale = args.scale
    epochs = args.epochs
    batch = args.batch
    finetune = args.finetune
    learning_rate = args.lr
    load_flag = args.fromscratch
    traindir = args.traindir
    general100_dir = args.general100_dir 
    test_image = args.image

    dataset_path = traindir
    augmented_path = "./augmented"

    # Set checkpoint paths for different scales and models
    ckpt_path = ""
    if scale == 2:
        ckpt_path = "./CKPT_dir/x2/"
    elif scale == 3:
        ckpt_path = "./CKPT_dir/x3/"
    elif scale == 4:
        ckpt_path = "./CKPT_dir/x4/"
    else:
        print("Upscale factor scale is not supported. Choose 2, 3 or 4.")
        exit()
    
    # Set gpu 
    config = tf.ConfigProto() #log_device_placement=True
    config.gpu_options.allow_growth = True

    # Dynamic placeholders
    x = tf.placeholder(tf.float32, [None, None, None, 3], name='x_placeholder')
    y = tf.placeholder(tf.float32, [None, None, None, 3], name='y_placeholder')
    y_shape = tf.shape(y, name='y_shape')
    
    # -- Model
    # construct model
    out, loss, train_op, psnr = edsr.model(x, y, B, F, y_shape, scale, batch, learning_rate)

    # Create run instance
    run = run_utils.run(config, ckpt_path, x, y)

    if args.train:
        # If finetune, load model and train on general100
        if finetune:
            dataset_path = general100_dir
            augmented_path = "./augmented_general100"

        # Augment and then load train images
        utils.augment(dataset_path, save_path=augmented_path)
        all_image_paths = utils.getpaths(augmented_path)
        X_np, Y_np = utils.load_images(all_image_paths, scale)

        # List of test images
        test_X, test_Y = utils.getTestImages("/home/weber/Documents/gsoc/datasets/BSDS100", 10, scale)

        model_outputs = out, loss, train_op, psnr

        # Train
        run.train(X_np, Y_np, test_X, test_Y, epochs, batch, load_flag, model_outputs)

    if args.test:
        # Test image
        run.test_compare(test_image, out, scale)

    if args.export:
        run.export(scale)
    print("I ran successfully.")