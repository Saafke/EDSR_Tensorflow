import pathlib
import os
from PIL import Image
import numpy as np
import cv2 
import tensorflow as tf
import imutils #rotating images properly
import shutil
import random

def getTestImages(imagefolder, nr, scale):
    
    test_x = list()
    test_y = list()

    random.seed(30)
    filenames = random.sample(os.listdir(imagefolder), nr)
    for fname in filenames:
        srcpath = os.path.join(imagefolder, fname)

        im = np.array(Image.open(srcpath))
        
        # make divisible by scale
        hr = im[0:(im.shape[0] - (im.shape[0] % scale)),
                           0:(im.shape[1] - (im.shape[1] % scale)), :]
        lr = cv2.resize(hr, (int(hr.shape[1] / scale), int(hr.shape[0] / scale)), 
                           interpolation=cv2.INTER_CUBIC)

        test_x.append(lr / 255.0)
        test_y.append(hr)
    return test_x, test_y

# # -- Code: used for calculating mean and std per image channel
def calculate_mean(paths):
    # -- Loop through all images
    total_means = [0, 0, 0]
    total_stand = [0, 0, 0]
    im_counter = 0

    for p in paths:
        im_counter += 1

        # -- Get the image.
        image = np.asarray(Image.open(p))
        #image = image / 255

        # -- Compute means.
        means_three = np.mean(image, axis=(0, 1), dtype=np.float64)

        # -- Compute std.
        std_three = np.std(image, axis=(0, 1), dtype=np.float64)

        # -- Print every ...
        if im_counter % 1000 == 0:
            print("mean:", means_three)
            print("std:", std_three)

        # -- Update total.
        total_means += means_three
        total_stand += std_three

    # -- Divide by the total to get the average.
    total_means /= im_counter
    total_stand /= im_counter

    total_means /= 255
    total_stand /= 255
    # -- Print result.
    print("total means:", total_means)
    print("total stand:", total_stand)

def getpaths(path):
    """
    Get all image paths from folder 'path'
    """
    data = pathlib.Path(path)
    all_image_paths = list(data.glob('*'))
    all_image_paths = [str(p) for p in all_image_paths]
    return all_image_paths

def augment(dataset_path, save_path):
    if(not os.path.isdir(save_path)):
        print("Making augmented images...")
        os.mkdir(save_path)

        do_augmentations(dataset_path, save_path)
        
        #count new images
        save_path, dirs, files = next(os.walk(save_path))
        file_count = len(files)
        
        print("{} augmented images are stored in the folder {}".format(file_count, save_path))

def rotate(img):
    """
    Function that rotates an image 90 degrees 4 times.

    returns:
    4 image arrays each rotated 90 degrees
    """
    rotated90 = imutils.rotate_bound(img, 90)
    rotated180 = imutils.rotate_bound(img, 180)
    rotated270 = imutils.rotate_bound(img, 270)

    return img, rotated90, rotated180, rotated270

def downscale(img):
    """
    Downscales an image 0.9x, 0.8x, 0.7x and 0.6x.

    Returns:
    5 image arrays
    """
    (w, h) = img.shape[:2]
    img09 = cv2.resize(img, dsize=(int(h*0.9),int(w*0.9)), interpolation=cv2.INTER_CUBIC)
    img08 = cv2.resize(img, dsize=(int(h*0.8),int(w*0.8)), interpolation=cv2.INTER_CUBIC)
    img07 = cv2.resize(img, dsize=(int(h*0.7),int(w*0.7)), interpolation=cv2.INTER_CUBIC)
    img06 = cv2.resize(img, dsize=(int(h*0.6),int(w*0.6)), interpolation=cv2.INTER_CUBIC)

    return img, img09, img08, img07, img06

def augment_image(img):
    """
    Rotates. Creates 20x images.
    """
    augmented_images = []

    rotated_images = rotate(img)
    
    for im in rotated_images:
        augmented_images.append(im)

    return augmented_images

def do_augmentations(dataset_path, save_path):
    """
    Does augmentations on all images in folder 'path'.
    """
    # get all image paths from folder
    dir = pathlib.Path(dataset_path)
    all_image_paths = list(dir.glob('*'))
    all_image_paths = [str(x) for x in all_image_paths]

    im_counter = 0
    # do augmentations
    for path in all_image_paths:
        # open current image as array
        img = Image.open(path)
        img = np.array(img)

        augm_counter = 0
        # get augmented images
        augmented_images = augment_image(img)
        for im in augmented_images: #save them all to ./augmented
            x = Image.fromarray(im)
            x.save(save_path + "/img{}aug{}.png".format(im_counter, augm_counter))
            augm_counter += 1
        im_counter += 1


def make_dataset(upscale_factor, batch):
    """
    Creates the superresolution patches placeholder dataset.
    """
    # make placeholders
    training_batches = tf.placeholder_with_default(tf.constant(batch, dtype=tf.int64), shape=[], name="batch_size_input")
    path_inputs = tf.placeholder(tf.string, shape=[None])

    # make dataset
    path_dataset = tf.data.Dataset.from_tensor_slices(path_inputs)
    train_dataset = path_dataset.flat_map(lambda x: data_tensor_slices(load_image(x, upscale_factor)))
    train_dataset = train_dataset.shuffle(buffer_size=(1820)) #91-image dataset times augmentations
    train_dataset = train_dataset.batch(training_batches)

    return path_inputs, training_batches, train_dataset

def data_tensor_slices((x,y)):
    """
    Returns a tensorflow dataset from images patches.
    """
    return tf.data.Dataset.from_tensor_slices((x, y))

def load_images(paths, scale):
    """
    Loads all images into proper low res and corresponding high res patches.

    Returns:
    List x and y
    """
    print("Loading images into patches...")
    # init lists
    x = []
    y = []
    
    # set lr and hr sizes
    lr_size = 48
    hr_size = lr_size * scale
    
    # loop over all image paths
    for p in paths:
        # -- Loading image 
        im = cv2.imread(p)
        #im = np.asarray(Image.open(p))
        # Normalize
        # mean rgb - 0.57880239, 0.45851252, 0.33622327
        # sdev rgb - 0.21396621, 0.1942685 , 0.20073332
        im = im / 255.0
        # -- Creating LR and HR images
        # make current image divisible by scale (because current image is the HR image)
        hr = im[0:(im.shape[0] - (im.shape[0] % scale)),
                           0:(im.shape[1] - (im.shape[1] % scale)), :]
        lr = cv2.resize(hr, (int(hr.shape[1] / scale), int(hr.shape[0] / scale)), 
                           interpolation=cv2.INTER_CUBIC)

        # -- Extract patches from the images 
        # TODO: overlapping pixels
        # TODO: don't round down, i.e. get all patches
        numx = int(lr.shape[0] / lr_size)
        numy = int(lr.shape[1] / lr_size)

        for i in range(0, numx):
            startx = i * lr_size
            endx = (i * lr_size) + lr_size
            
            startx_hr = i * hr_size
            endx_hr = (i * hr_size) + hr_size
            
            for j in range(0, numy):
                starty = j * lr_size
                endy = (j * lr_size) + lr_size
                
                starty_hr = j * hr_size
                endy_hr = (j * hr_size) + hr_size

                crop_lr = lr[startx:endx, starty:endy]
                crop_hr = hr[startx_hr:endx_hr, starty_hr:endy_hr]

                label = crop_hr.reshape(((hr_size), (hr_size), 3))
                inpt = crop_lr.reshape((lr_size, lr_size, 3))
                
                #label = np.expand_dims(label, axis=0)
                #inpt = np.expand_dims(inpt, axis=0)

                x.append(inpt)
                y.append(label)
    print("Images loaded.")
    return x, y

def load_image_old(path, scale):
    """
    Loads an image into proper low res and corresponding high res patches.
    """
    # init
    lr_size = 48

    #read tf image 
    im = tf.read_file(path)
    im = tf.image.decode_png(im, channels=3)
    im = tf.cast(im, tf.float32) / 255.0

    # make dimensions divisible by scale and make hr shape
    X = tf.dtypes.cast((tf.shape(im)[0] / scale), dtype=tf.int32) * scale
    Y = tf.dtypes.cast((tf.shape(im)[1] / scale), dtype=tf.int32) * scale
    high = tf.image.crop_to_bounding_box(im, 0, 0, X, Y)
    print("high.shape: ")
    print(high.shape)
    print("\n")

    # make lr shape
    imgshape = tf.shape(high)
    size = [imgshape[0] / scale, imgshape[1] / scale]
    low = tf.image.resize_images(high, size=size, method=tf.image.ResizeMethod.BILINEAR)
    print("low.shape: ")
    print(low.shape)
    print("\n")

    hshape = tf.shape(high)
    lshape = tf.shape(low)

    # make it 4d (1, h, w, channels)
    low_r = tf.reshape(low, [1, lshape[0], lshape[1], 3])
    high_r = tf.reshape(high, [1, hshape[0], hshape[1], 3])
    print("low_r.shape: ")
    print(low_r.shape)
    print("high_r.shape: ")
    print(high_r.shape)
    print("\n")

    # get image patches (size depending on scale)
    # - ksizes  = The size of the sliding window for each dimension of images.
    # - strides = How far the centers of two consecutive patches are in the images.
    # - rates   = This is the input stride, specifying how far two consecutive patch samples are in the input.
    # TODO: SHOULD BE ONE PIXEL OVERLAPPING
    slice_l = tf.image.extract_image_patches(low_r, ksizes=[1, lr_size, lr_size, 1], strides=[1, lr_size, lr_size , 1], rates=[1, 1, 1, 1], padding="VALID")
    slice_h = tf.image.extract_image_patches(high_r, ksizes=[1, lr_size * scale, lr_size * scale, 1], strides=[1, lr_size * scale, lr_size * scale, 1],
                                                rates=[1, 1, 1, 1], padding="VALID")
    print("slice_l.shape: ")
    print(slice_l.shape)
    print("slice_h.shape: ")
    print(slice_h.shape)
    print("\n")

    #reshape patches to be in the shape (amount_of_patches, height, weight, channels)
    LR_image_patches = tf.reshape(slice_l, [tf.shape(slice_l)[1] * tf.shape(slice_l)[2], lr_h, lr_w, channels])
    HR_image_patches = tf.reshape(slice_h, [tf.shape(slice_h)[1] * tf.shape(slice_h)[2], lr_h * scale, lr_w * scale, channels])
    print("LR_image_patches.shape: ")
    print(LR_image_patches.shape)
    print("HR_image_patches.shape: ")
    print(HR_image_patches.shape)
    print("\n")

    return (LR_image_patches, HR_image_patches)
    #return tf.data.Dataset.from_tensor_slices((LR_image_patches, HR_image_patches))


def make_dataset(scale, batch):
    """
    Creates the superresolution patches placeholder dataset.
    """
    # make placeholders
    training_batches = tf.placeholder_with_default(tf.constant(batch, dtype=tf.int64), shape=[], name="batch_size_input")
    path_inputs = tf.placeholder(tf.string, shape=[None])

    # make dataset
    path_dataset = tf.data.Dataset.from_tensor_slices(path_inputs)
    train_dataset = path_dataset.flat_map(lambda x: data_tensor_slices(load_image(x, scale)))
    #train_dataset = train_dataset.shuffle(buffer_size=(1820)) #91-image dataset times augmentations
    train_dataset = train_dataset.batch(training_batches)

    return path_inputs, training_batches, train_dataset

def data_tensor_slices((x,y)):
    """
    Returns a tensorflow dataset from images patches.
    """
    return tf.data.Dataset.from_tensor_slices((x, y))
