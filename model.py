# Deep Learning for Contrast-Enhanced T1 MR Image Synthesis
# @author: Alexander F.I. Osman, April 2023

"""
PART I: TRAINING THE MODEL

This code demonstrates a 3D Res U-Net architecture for contrast-enhanced MR image
synthesis from contrast-free image.

Architectures: 3D U-Net

Dataset: BRATS'2021 challenge dataset.

The training process goes through the following steps:
1. Load the data
2. Pre-process the data (clean the data, resize, normalize, etc.)
3. Build the model architecture (3D U-Net)
4. Train and validate the model for image translations
"""

###############################################################################
# 1. LOADING A SAMPLE DATA SET AND VISUALIZE ##################################
###############################################################################

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
import random
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# Initial understanding of the dataset.
# Read source images (T1)
dataset_path = 'E:/Datasets/BRATS_2021_Challenge/BraTS2021_TrainingData/'
img_t1 = (dataset_path + 'BraTS2021_00000/BraTS2021_00000_t1.nii.gz')
img_t1 = nib.load(img_t1).get_fdata()
img_t1 = np.rot90(np.array(img_t1), k=3)

# Read source images (T2)
dataset_path = 'E:/Datasets/BRATS_2021_Challenge/BraTS2021_TrainingData/'
img_t2 = (dataset_path + 'BraTS2021_00000/BraTS2021_00000_t2.nii.gz')
img_t2 = nib.load(img_t2).get_fdata()
img_t2 = np.rot90(np.array(img_t2), k=3)

# Read source images (FLAIR)
dataset_path = 'E:/Datasets/BRATS_2021_Challenge/BraTS2021_TrainingData/'
img_fl = (dataset_path + 'BraTS2021_00000/BraTS2021_00000_flair.nii.gz')
img_fl = nib.load(img_fl).get_fdata()
img_fl = np.rot90(np.array(img_fl), k=3)

# Read target images (T1ce)
dataset_path = 'E:/Datasets/BRATS_2021_Challenge/BraTS2021_TrainingData/'
img_t1ce = (dataset_path + 'BraTS2021_00000/BraTS2021_00000_t1ce.nii.gz')
img_t1ce = nib.load(img_t1ce).get_fdata()
img_t1ce = np.rot90(np.array(img_t1ce), k=3)

print("Used memory to store img_sc: ", img_t1.nbytes/(1024*1024), "MB")
print("Used memory to store img_sc: ", img_t2.nbytes/(1024*1024), "MB")
print("Used memory to store img_sc: ", img_fl.nbytes/(1024*1024), "MB")
print("Used memory to store img_tg: ", img_t1ce.nbytes/(1024*1024), "MB")

# Plot
slice_numb = 82
#slice_numb = random.randint(0, img_t1.shape[2])
plt.figure(figsize=(12, 8))
plt.subplot(221)
plt.imshow(img_t1[:,:,slice_numb], cmap='gray')
plt.colorbar(), plt.title('T1 image'), plt.axis('tight')
plt.subplot(222)
plt.imshow(img_t2[:,:,slice_numb], cmap='gray')
plt.colorbar(), plt.title('T2 image'), plt.axis('tight')
plt.subplot(223)
plt.imshow(img_fl[:,:,slice_numb], cmap='gray')
plt.colorbar(), plt.title('FLAIR image'), plt.axis('tight')
plt.subplot(224)
plt.imshow(img_t1ce[:,:,slice_numb], aspect=0.5, cmap='gray')
plt.colorbar(), plt.title('T1ce image'), plt.axis('tight')
plt.show()


###############################################################################
# 2. DATA PREPROCESSING AND SAVING FILES ######################################
###############################################################################

import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage.transform import resize
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
import glob
import splitfolders
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def read_nifti_file(filepath):
    """Read and load volume"""
    scan = nib.load(filepath).get_fdata()
    return scan

def normalize_image_volume(volume):
    """Normalize the volume"""
    volume = (volume - volume.mean()) / volume.std()
    volume = (volume - volume.min()) / (volume.max() - volume.min())
    return volume.astype("float32")

def crop_image_volume(volume):
    """Crop across x, y axes"""
    volume = volume[30:210, 30:210, 13:141]   # 180x180x128
    return volume

def resize_image_volume(volume):
    """Resize across z-axis"""
    # Set the desired depth
    desired_width, desired_height, desired_depth = 128, 128, 128
    # Get current depth
    current_width, current_height, current_depth = volume.shape[0], volume.shape[1], volume.shape[2]
    # Compute depth factor
    width = current_width / desired_width
    height = current_height / desired_height
    depth = current_depth / desired_depth
    width_factor = 1 / width
    height_factor = 1 / height
    depth_factor = 1 / depth
    # Rotate
    volume = np.rot90(np.array(volume), k=3)
    # Resize across z-axis
    volume = ndimage.zoom(volume, (width_factor, height_factor, depth_factor), order=1)
    return volume


# Process the data (crop, normalize, and split)
t1_list = sorted(glob.glob('E:/Datasets/BRATS_2021_Challenge/BraTS2021_TrainingData/*/*t1.nii.gz'))
t2_list = sorted(glob.glob('E:/Datasets/BRATS_2021_Challenge/BraTS2021_TrainingData/*/*t2.nii.gz'))
fl_list = sorted(glob.glob('E:/Datasets/BRATS_2021_Challenge/BraTS2021_TrainingData/*/*flair.nii.gz'))
t1ce_list = sorted(glob.glob('E:/Datasets/BRATS_2021_Challenge/BraTS2021_TrainingData/*/*t1ce.nii.gz'))

for item in tqdm(range(len(t1_list)), desc='Processing and saving images'):
    # IMAGE DATA
    # T1 images
    temp_img_t1 = read_nifti_file(t1_list[item])
    temp_img_t1 = crop_image_volume(temp_img_t1)
    temp_img_t1 = resize_image_volume(temp_img_t1)
    temp_img_t1 = normalize_image_volume(temp_img_t1)

    # T2 images
    temp_img_t2 = read_nifti_file(t2_list[item])
    temp_img_t2 = crop_image_volume(temp_img_t2)
    temp_img_t2 = resize_image_volume(temp_img_t2)
    temp_img_t2 = normalize_image_volume(temp_img_t2)

    # FLAIR images
    temp_img_fl = read_nifti_file(fl_list[item])
    temp_img_fl = crop_image_volume(temp_img_fl)
    temp_img_fl = resize_image_volume(temp_img_fl)
    temp_img_fl = normalize_image_volume(temp_img_fl)

    # Combine/merge images
    temp_img_t1_t2_fl = np.stack([temp_img_t1, temp_img_t2, temp_img_fl], axis=3)

    # T1ce images
    temp_img_t1ce = read_nifti_file(t1ce_list[item])
    temp_img_t1ce = crop_image_volume(temp_img_t1ce)
    temp_img_t1ce = resize_image_volume(temp_img_t1ce)
    temp_img_t1ce = normalize_image_volume(temp_img_t1ce)
    temp_img_t1ce = np.expand_dims(temp_img_t1ce, -1)

    # Save images
    np.save('E:/Datasets/BRATS_2021_Challenge/saved_dataset_comb/images_t1_t2_fl/image_' + str(item) + '.npy',
                temp_img_t1_t2_fl)
    np.save('E:/Datasets/BRATS_2021_Challenge/saved_dataset_comb/images_t1ce/image_' + str(item) + '.npy',
                temp_img_t1ce)


""" When use trained U-Net, it deals with 3-channel
# #Convert grey image to 3 channels by copying channel 3 times.
# We do this as our U-Net model expects 3 channel input.
train_img = np.stack((input_img,)*3, axis=-1)
train_mask = np.expand_dims(input_mask, axis=4)
train_mask_cat = to_categorical(train_mask, num_classes=n_classes)
"""

# Repeat the same from above for validation data folder OR split training data
# into train, val, and test folders. The created folders will be used for semantic
# seg using datagens.

input_folder = 'E:/Datasets/BRATS_2021_Challenge/saved_dataset_comb/'
output_folder = 'E:/Datasets/BRATS_2021_Challenge/train_val_test_datasets_comb/'

# Split into training and validation set
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(0.64, 0.16, 0.20), group_prefix=None)

"""
from patchify import patchify, unpatchify 
#Here we load 256x256x256 pixel volume. We will break it into patches of 64x64x64 
# for training.
img_patches = patchify(image, (64, 64, 64), step=64)  #Step=64 for 64 patches means no overlap
"""


###############################################################################
# 3. LOAD THE PREPROCESSED SAVED DATA #########################################
###############################################################################

import numpy as np
import os
import glob
import pandas as pd
import matplotlib
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
import random
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# Load the processed and saved data and visualize for sanity check.
train_img_t1_t2_fl_dir = "E:/Datasets/BRATS_2021_Challenge/train_val_test_datasets_comb/train/images_t1_t2_fl/"
train_img_t1ce_dir = "E:/Datasets/BRATS_2021_Challenge/train_val_test_datasets_comb/train/images_t1ce/"

train_img_t1_t2_fl_list = os.listdir(train_img_t1_t2_fl_dir)
train_img_t1ce_list = os.listdir(train_img_t1ce_dir)

num_images = len(os.listdir(train_img_t1ce_dir))
# Load a single image
img_num = random.randint(0, num_images-1)
test_img_t1_t2_fl = np.load(train_img_t1_t2_fl_dir+train_img_t1_t2_fl_list[img_num])
test_img_t1ce = np.load(train_img_t1ce_dir+train_img_t1ce_list[img_num])

print("Used memory to store test_img_t1_t2_fl: ", test_img_t1_t2_fl.nbytes/(1024*1024), "MB")
print("Used memory to store test_img_t1ce: ", test_img_t1ce.nbytes/(1024*1024), "MB")

# Plot
slice_numb = random.randint(0, test_img_t1ce.shape[2])
plt.figure(figsize=(12, 8))
plt.subplot(221)
plt.imshow(test_img_t1_t2_fl[:,:,slice_numb,0], cmap='gray')
plt.colorbar(), plt.title('T1 image'), plt.axis('tight'), plt.axis('off')
plt.subplot(222)
plt.imshow(test_img_t1_t2_fl[:,:,slice_numb,1], cmap='gray')
plt.colorbar(), plt.title('T2 image'), plt.axis('tight'), plt.axis('off')
plt.subplot(223)
plt.imshow(test_img_t1_t2_fl[:,:,slice_numb,2], cmap='gray')
plt.colorbar(), plt.title('FLAIR image'), plt.axis('tight'), plt.axis('off')
plt.subplot(224)
plt.imshow(test_img_t1ce[:,:,slice_numb], cmap='gray')
plt.colorbar(), plt.title('T1ce image'), plt.axis('tight'), plt.axis('off')
plt.show()


###############################################################################
# 4. DEFINE DATA GENERATOR ####################################################
###############################################################################

import numpy as np
import random
import matplotlib
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def load_img(img_dir, img_list):
    images = []
    for i, image_name in enumerate(img_list):
        if image_name.split('.')[1] == 'npy':
            image = np.load(img_dir+image_name)
            images.append(image)
    images = np.array(images)
    return images


def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):
    L = len(img_list)
    # keras needs the generator infinite, so we will use while true
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end, L)
            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])
            yield (X,Y) # a tuple with two numpy arrays with batch_size samples
            batch_start += batch_size
            batch_end += batch_size


# Define the image generators for training and validation
train_img_t1_t2_fl_dir = "E:/Datasets/BRATS_2021_Challenge/train_val_test_datasets_comb/train/images_t1_t2_fl/"
train_img_t1ce_dir = "E:/Datasets/BRATS_2021_Challenge/train_val_test_datasets_comb/train/images_t1ce/"

val_img_t1_t2_fl_dir = "E:/Datasets/BRATS_2021_Challenge/train_val_test_datasets_comb/val/images_t1_t2_fl/"
val_img_t1ce_dir = "E:/Datasets/BRATS_2021_Challenge/train_val_test_datasets_comb/val/images_t1ce/"

train_img_t1_t2_fl_list = os.listdir(train_img_t1_t2_fl_dir)
train_img_t1ce_list = os.listdir(train_img_t1ce_dir)

val_img_t1_t2_fl_list = os.listdir(val_img_t1_t2_fl_dir)
val_img_t1ce_list = os.listdir(val_img_t1ce_dir)

batch_size = 1
train_img_datagen = imageLoader(train_img_t1_t2_fl_dir, train_img_t1_t2_fl_list,
                                train_img_t1ce_dir, train_img_t1ce_list, batch_size)
val_img_datagen = imageLoader(val_img_t1_t2_fl_dir, val_img_t1_t2_fl_list, val_img_t1ce_dir,
                              val_img_t1ce_list, batch_size)

# Verify generator
img_t1_t2_fl, img_t1ce = train_img_datagen.__next__()

img_num = random.randint(0, img_t1ce.shape[0]-1)
test_img_t1_t2_fl = img_t1_t2_fl[img_num]
test_img_t1ce = img_t1ce[img_num]

# Plot
slice_numb = random.randint(0, test_img_t1ce.shape[2])
plt.figure(figsize=(12, 8))
plt.subplot(221)
plt.imshow(test_img_t1_t2_fl[:,:,slice_numb,0], cmap='gray')
plt.colorbar(), plt.title('T1 image'), plt.axis('tight'), plt.axis('off')
plt.subplot(222)
plt.imshow(test_img_t1_t2_fl[:,:,slice_numb,1], cmap='gray')
plt.colorbar(), plt.title('T2 image'), plt.axis('tight'), plt.axis('off')
plt.subplot(223)
plt.imshow(test_img_t1_t2_fl[:,:,slice_numb,2], cmap='gray')
plt.colorbar(), plt.title('FLAIR image'), plt.axis('tight'), plt.axis('off')
plt.subplot(224)
plt.imshow(test_img_t1ce[:,:,slice_numb], cmap='gray')
plt.colorbar(), plt.title('T1ce image'), plt.axis('tight'), plt.axis('off')
plt.show()


###############################################################################
# 5. BUILD THE MODEL ARCHITECTURE #############################################
###############################################################################

from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, UpSampling3D, BatchNormalization, Dropout, Activation, add
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# 3D U-Net
def build_3DUNet_model(img_height, img_width, img_depth, img_channels):
    """ 3D Standard U-NET Architecture
     :param input_shape: (image height, image width, image depth, image channels)
     :return: model
     """
    inputs = Input((img_height, img_width, img_depth, img_channels))
    ini_numb_of_filters = 16
    s = inputs

    """ Contraction path """
    c1 = Conv3D(ini_numb_of_filters, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(s)
    c1 = Dropout(0.10)(c1)
    c1 = Conv3D(ini_numb_of_filters, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(c1)
    p1 = MaxPooling3D(pool_size=(2, 2, 2))(c1)

    c2 = Conv3D(ini_numb_of_filters * 2, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(p1)
    c2 = Dropout(0.15)(c2)
    c2 = Conv3D(ini_numb_of_filters * 2, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(c2)
    p2 = MaxPooling3D(pool_size=(2, 2, 2))(c2)

    c3 = Conv3D(ini_numb_of_filters * 4, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(p2)
    c3 = Dropout(0.20)(c3)
    c3 = Conv3D(ini_numb_of_filters * 4, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(c3)
    p3 = MaxPooling3D(pool_size=(2, 2, 2))(c3)

    c4 = Conv3D(ini_numb_of_filters * 8, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(p3)
    c4 = Dropout(0.25)(c4)
    c4 = Conv3D(ini_numb_of_filters * 8, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(c4)
    p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)

    """ Bridge """
    c5 = Conv3D(ini_numb_of_filters * 16, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(p4)
    c5 = Dropout(0.30)(c5)

    c5 = Conv3D(ini_numb_of_filters * 16, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(c5)

    """ Expansive path """
    # u6 = UpSampling3D(size=(2, 2, 2), data_format="channels_last")(c5)
    u6 = Conv3DTranspose(ini_numb_of_filters * 8, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv3D(ini_numb_of_filters * 8, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(u6)
    c6 = Dropout(0.25)(c6)
    c6 = Conv3D(ini_numb_of_filters * 8, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(c6)

    # u7 = UpSampling3D(size=(2, 2, 2), data_format="channels_last")(c6)
    u7 = Conv3DTranspose(ini_numb_of_filters * 4, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv3D(ini_numb_of_filters * 4, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(u7)
    c7 = Dropout(0.20)(c7)
    c7 = Conv3D(ini_numb_of_filters * 4, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(c7)

    # u8 = UpSampling3D(size=(2, 2, 2), data_format="channels_last")(c7)
    u8 = Conv3DTranspose(ini_numb_of_filters * 2, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv3D(ini_numb_of_filters * 2, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(u8)
    c8 = Dropout(0.15)(c8)
    c8 = Conv3D(ini_numb_of_filters * 2, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(c8)

    # u9 = UpSampling3D(size=(2, 2, 2), data_format="channels_last")(c8)
    u9 = Conv3DTranspose(ini_numb_of_filters, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv3D(ini_numb_of_filters, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(u9)
    c9 = Dropout(0.10)(c9)
    c9 = Conv3D(ini_numb_of_filters, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(c9)

    outputs = Conv3D(1, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='relu')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    # compile model outside of this function to make it flexible.
    model.summary()
    return model


# Test if everything is working ok.
model = build_3DUNet_model(128, 128, 128, 3)
print(model.input_shape)
print(model.output_shape)


# Dilated U-Net (dilated in bottleneck only)
def build_dilated_3DUNet_model(img_height, img_width, img_depth, img_channels):
    """ 3D Standard U-NET Architecture
     :param input_shape: (image height, image width, image depth, image channels)
     :return: model
     """
    inputs = Input((img_height, img_width, img_depth, img_channels))
    ini_numb_of_filters = 16
    s = inputs

    """ Contraction path """
    c1 = Conv3D(ini_numb_of_filters, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(s)
    c1 = Dropout(0.10)(c1)
    c1 = Conv3D(ini_numb_of_filters, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(c1)
    p1 = MaxPooling3D(pool_size=(2, 2, 2))(c1)

    c2 = Conv3D(ini_numb_of_filters * 2, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(p1)
    c2 = Dropout(0.15)(c2)
    c2 = Conv3D(ini_numb_of_filters * 2, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(c2)
    p2 = MaxPooling3D(pool_size=(2, 2, 2))(c2)

    c3 = Conv3D(ini_numb_of_filters * 4, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(p2)
    c3 = Dropout(0.20)(c3)
    c3 = Conv3D(ini_numb_of_filters * 4, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(c3)
    p3 = MaxPooling3D(pool_size=(2, 2, 2))(c3)

    c4 = Conv3D(ini_numb_of_filters * 8, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(p3)
    c4 = Dropout(0.25)(c4)
    c4 = Conv3D(ini_numb_of_filters * 8, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(c4)
    p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)

    """ Bridge """
    c51 = Conv3D(ini_numb_of_filters * 16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                 dilation_rate=(1, 1, 1), activation='relu', kernel_initializer='he_uniform')(p4)
    c51 = Dropout(0.30)(c51)

    c52 = Conv3D(ini_numb_of_filters * 16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                 dilation_rate=(2, 2, 2), activation='relu', kernel_initializer='he_uniform')(c51)
    c52 = Dropout(0.30)(c52)

    c55 = Conv3D(ini_numb_of_filters * 16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                 dilation_rate=(5, 5, 5), activation='relu', kernel_initializer='he_uniform')(c52)
    c55 = Dropout(0.30)(c55)

    c57 = Conv3D(ini_numb_of_filters * 16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                 dilation_rate=(7, 7, 7), activation='relu', kernel_initializer='he_uniform')(c55)
    c57 = Dropout(0.30)(c57)

    c59 = Conv3D(ini_numb_of_filters * 16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                 dilation_rate=(9, 9, 9), activation='relu', kernel_initializer='he_uniform')(c57)
    c59 = Dropout(0.30)(c59)

    """ Expansive path """
    # u6 = UpSampling3D(size=(2, 2, 2), data_format="channels_last")(c59)
    u6 = Conv3DTranspose(ini_numb_of_filters * 8, (2, 2, 2), strides=(2, 2, 2), padding='same')(c59)
    u6 = concatenate([u6, c4])
    c6 = Conv3D(ini_numb_of_filters * 8, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(u6)
    c6 = Dropout(0.25)(c6)
    c6 = Conv3D(ini_numb_of_filters * 8, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(c6)

    # u7 = UpSampling3D(size=(2, 2, 2), data_format="channels_last")(c6)
    u7 = Conv3DTranspose(ini_numb_of_filters * 4, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv3D(ini_numb_of_filters * 4, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(u7)
    c7 = Dropout(0.20)(c7)
    c7 = Conv3D(ini_numb_of_filters * 4, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(c7)

    # u8 = UpSampling3D(size=(2, 2, 2), data_format="channels_last")(c7)
    u8 = Conv3DTranspose(ini_numb_of_filters * 2, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv3D(ini_numb_of_filters * 2, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(u8)
    c8 = Dropout(0.15)(c8)
    c8 = Conv3D(ini_numb_of_filters * 2, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(c8)

    # u9 = UpSampling3D(size=(2, 2, 2), data_format="channels_last")(c8)
    u9 = Conv3DTranspose(ini_numb_of_filters, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv3D(ini_numb_of_filters, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(u9)
    c9 = Dropout(0.10)(c9)
    c9 = Conv3D(ini_numb_of_filters, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(c9)

    outputs = Conv3D(1, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='relu')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    # compile model outside of this function to make it flexible.
    model.summary()
    return model


# Test if everything is working ok.
model = build_dilated_3DUNet_model(128, 128, 128, 3)
print(model.input_shape)
print(model.output_shape)


# Dense-dilated U-Net (dilated in bottleneck only) == 3 levels (Med Phys Paper)
def build_dilated_3DUNet_model(img_height, img_width, img_depth, img_channels):
    """ 3D Standard U-NET Architecture
     :param input_shape: (image height, image width, image depth, image channels)
     :return: model
     """
    inputs = Input((img_height, img_width, img_depth, img_channels))
    ini_numb_of_filters = 16
    s = inputs

    """ Contraction path """
    c1 = Conv3D(ini_numb_of_filters, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(s)
    c1 = Dropout(0.10)(c1)
    c1 = Conv3D(ini_numb_of_filters, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(c1)
    p1 = MaxPooling3D(pool_size=(2, 2, 2))(c1)

    c2 = Conv3D(ini_numb_of_filters * 2, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(p1)
    c2 = Dropout(0.15)(c2)
    c2 = Conv3D(ini_numb_of_filters * 2, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(c2)
    p2 = MaxPooling3D(pool_size=(2, 2, 2))(c2)

    c3 = Conv3D(ini_numb_of_filters * 4, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(p2)
    c3 = Dropout(0.20)(c3)
    c3 = Conv3D(ini_numb_of_filters * 4, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(c3)
    p3 = MaxPooling3D(pool_size=(2, 2, 2))(c3)

    """ Bridge """
    c51 = Conv3D(ini_numb_of_filters * 8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                 dilation_rate=(1, 1, 1), activation='relu', kernel_initializer='he_uniform')(p3)
    c51 = Dropout(0.25)(c51)

    c52 = Conv3D(ini_numb_of_filters * 8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                 dilation_rate=(2, 2, 2), activation='relu', kernel_initializer='he_uniform')(c51)
    c52 = Dropout(0.25)(c52)

    conc = concatenate([c51, c52])
    c55 = Conv3D(ini_numb_of_filters * 8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                 dilation_rate=(5, 5, 5), activation='relu', kernel_initializer='he_uniform')(conc)
    c55 = Dropout(0.25)(c55)

    """ Expansive path """
    # u7 = UpSampling3D(size=(2, 2, 2), data_format="channels_last")(c55)
    u7 = Conv3DTranspose(ini_numb_of_filters * 4, (2, 2, 2), strides=(2, 2, 2), padding='same')(c55)
    u7 = concatenate([u7, c3])
    c7 = Conv3D(ini_numb_of_filters * 4, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(u7)
    c7 = Dropout(0.20)(c7)
    c7 = Conv3D(ini_numb_of_filters * 4, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(c7)

    # u8 = UpSampling3D(size=(2, 2, 2), data_format="channels_last")(c7)
    u8 = Conv3DTranspose(ini_numb_of_filters * 2, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv3D(ini_numb_of_filters * 2, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(u8)
    c8 = Dropout(0.15)(c8)
    c8 = Conv3D(ini_numb_of_filters * 2, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(c8)

    # u9 = UpSampling3D(size=(2, 2, 2), data_format="channels_last")(c8)
    u9 = Conv3DTranspose(ini_numb_of_filters, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv3D(ini_numb_of_filters, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(u9)
    c9 = Dropout(0.10)(c9)
    c9 = Conv3D(ini_numb_of_filters, kernel_size=(3, 3, 3), strides=(1, 1, 1),
                padding='same', activation='relu', kernel_initializer='he_uniform')(c9)

    outputs = Conv3D(1, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='relu')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    # compile model outside of this function to make it flexible.
    model.summary()
    return model


# Test if everything is working ok.
model = build_dilated_3DUNet_model(128, 128, 128, 3)
print(model.input_shape)
print(model.output_shape)


# Residual U-Net
def res_conv_block(x, size, dropout):
    """ Residual convolutional layer """
    # Either put activation function before the addition with shortcut
    # or after the addition (which would be as proposed in the original resNet).
    # 1. conv-Activation-conv-Activation-shortcut-shortcut
    # 2. conv-Activation-conv-shortcut-shortcut-Activation
    conv = Conv3D(size, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                  kernel_initializer='he_uniform')(x)
    conv = Activation('relu')(conv)
    conv = Conv3D(size, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                  kernel_initializer='he_uniform')(conv)
    # conv = Activation('relu')(conv)    #Activation before addition with shortcut
    if dropout > 0:
        conv = Dropout(dropout)(conv)
    shortcut = Conv3D(size, kernel_size=(1, 1, 1), strides=(1, 1, 1),
                      padding='same', kernel_initializer='he_uniform')(x)
    res_path = add([shortcut, conv])
    res_path = Activation('relu')(res_path)  # Activation after addition with shortcut (Original residual block)
    return res_path


def build_res_3DUNet_model(img_height, img_width, img_depth, img_channels):
    """ 3D Residual U-NET Architecture
     :param input_shape: (image height, image width, image depth, image channels)
     :return: model
     """
    inputs = Input((img_height, img_width, img_depth, img_channels))
    ini_numb_of_filters = 16
    s = inputs

    """ Contraction path """
    c1 = res_conv_block(s, ini_numb_of_filters, dropout=0.10)
    p1 = MaxPooling3D(pool_size=(2, 2, 2))(c1)

    c2 = res_conv_block(p1, ini_numb_of_filters * 2, dropout=0.15)
    p2 = MaxPooling3D(pool_size=(2, 2, 2))(c2)

    c3 = res_conv_block(p2, ini_numb_of_filters * 4, dropout=0.20)
    p3 = MaxPooling3D(pool_size=(2, 2, 2))(c3)

    c4 = res_conv_block(p3, ini_numb_of_filters * 8, dropout=0.25)
    p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)

    """ Bridge """
    c5 = res_conv_block(p4, ini_numb_of_filters * 16, dropout=0.30)

    """ Expansive path """
    # u6 = UpSampling3D(size=(2, 2, 2), data_format="channels_last")(c5)
    u6 = Conv3DTranspose(ini_numb_of_filters * 8, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = res_conv_block(u6, ini_numb_of_filters * 8, dropout=0.25)

    # u7 = UpSampling3D(size=(2, 2, 2), data_format="channels_last")(c6)
    u7 = Conv3DTranspose(ini_numb_of_filters * 4, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = res_conv_block(u7, ini_numb_of_filters * 4, dropout=0.20)

    # u8 = UpSampling3D(size=(2, 2, 2), data_format="channels_last")(c7)
    u8 = Conv3DTranspose(2 * ini_numb_of_filters * 2, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = res_conv_block(u8, ini_numb_of_filters * 2, dropout=0.15)

    # u9 = UpSampling3D(size=(2, 2, 2), data_format="channels_last")(c8)
    u9 = Conv3DTranspose(ini_numb_of_filters, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = res_conv_block(u9, ini_numb_of_filters, dropout=0.10)

    """ Final convolutional layer """
    outputs = Conv3D(1, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='relu')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    # compile model outside of this function to make it flexible.
    model.summary()
    return model


# Test if everything is working ok.
model = build_res_3DUNet_model(128, 128, 128, 3)
print(model.input_shape)
print(model.output_shape)


# Dense_dilated-Residual U-Net
def res_conv_block(x, size, dropout):
    """ Residual convolutional layer """
    # Either put activation function before the addition with shortcut
    # or after the addition (which would be as proposed in the original resNet).
    # 1. conv-Activation-conv-Activation-shortcut-shortcut
    # 2. conv-Activation-conv-shortcut-shortcut-Activation
    conv = Conv3D(size, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                  kernel_initializer='he_uniform')(x)
    conv = Activation('relu')(conv)
    conv = Conv3D(size, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                  kernel_initializer='he_uniform')(conv)
    # conv = Activation('relu')(conv)    #Activation before addition with shortcut
    if dropout > 0:
        conv = Dropout(dropout)(conv)
    shortcut = Conv3D(size, kernel_size=(1, 1, 1), strides=(1, 1, 1),
                      padding='same', kernel_initializer='he_uniform')(x)
    res_path = add([shortcut, conv])
    res_path = Activation('relu')(res_path)  # Activation after addition with shortcut (Original residual block)
    return res_path


def build_res_dilated_3DUNet_model(img_height, img_width, img_depth, img_channels):
    """ 3D Residual U-NET Architecture
     :param input_shape: (image height, image width, image depth, image channels)
     :return: model
     """
    inputs = Input((img_height, img_width, img_depth, img_channels))
    ini_numb_of_filters = 16
    s = inputs

    """ Contraction path """
    c1 = res_conv_block(s, ini_numb_of_filters, dropout=0.10)
    p1 = MaxPooling3D(pool_size=(2, 2, 2))(c1)

    c2 = res_conv_block(p1, ini_numb_of_filters * 2, dropout=0.15)
    p2 = MaxPooling3D(pool_size=(2, 2, 2))(c2)

    c3 = res_conv_block(p2, ini_numb_of_filters * 4, dropout=0.20)
    p3 = MaxPooling3D(pool_size=(2, 2, 2))(c3)

    """ Bridge """
    c51 = Conv3D(ini_numb_of_filters * 8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                 dilation_rate=(1, 1, 1), activation='relu', kernel_initializer='he_uniform')(p3)
    c51 = Dropout(0.25)(c51)

    c52 = Conv3D(ini_numb_of_filters * 8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                 dilation_rate=(2, 2, 2), activation='relu', kernel_initializer='he_uniform')(c51)
    c52 = Dropout(0.25)(c52)

    conc = concatenate([c51, c52])
    c55 = Conv3D(ini_numb_of_filters * 8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                 dilation_rate=(5, 5, 5), activation='relu', kernel_initializer='he_uniform')(conc)
    c55 = Dropout(0.25)(c55)

    """ Expansive path """
    # u7 = UpSampling3D(size=(2, 2, 2), data_format="channels_last")(c55)
    u7 = Conv3DTranspose(ini_numb_of_filters * 4, (2, 2, 2), strides=(2, 2, 2), padding='same')(c55)
    u7 = concatenate([u7, c3])
    c7 = res_conv_block(u7, ini_numb_of_filters * 4, dropout=0.20)

    # u8 = UpSampling3D(size=(2, 2, 2), data_format="channels_last")(c7)
    u8 = Conv3DTranspose(2 * ini_numb_of_filters * 2, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = res_conv_block(u8, ini_numb_of_filters * 2, dropout=0.15)

    # u9 = UpSampling3D(size=(2, 2, 2), data_format="channels_last")(c8)
    u9 = Conv3DTranspose(ini_numb_of_filters, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = res_conv_block(u9, ini_numb_of_filters, dropout=0.10)

    """ Final convolutional layer """
    outputs = Conv3D(1, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='relu')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    # compile model outside of this function to make it flexible.
    model.summary()
    return model


# Test if everything is working ok.
model = build_res_dilated_3DUNet_model(128, 128, 128, 3)
print(model.input_shape)
print(model.output_shape)

###############################################################################
# 6. TRAINING THE MODEL #######################################################
###############################################################################

import time
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
import pandas as pd
import matplotlib
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.optimizers import SGD, RMSprop, Adam
import tensorflow as tf
import os
import keras
from keras import backend as keras
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def plot_learning_curve(filepath):
    df = pd.read_csv(filepath)
    df_x, df_yt, df_yv = df.values[:, 0], df.values[:, 2], df.values[:, 5]
    plt.figure(figsize=(5, 4))
    plt.plot(df_x, df_yt)
    plt.plot(df_x, df_yv)
    # plt.title('average training loss and validation loss')
    plt.ylabel('loss', fontsize=16)
    plt.xlabel('epoch', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(['training loss', 'validation loss'], fontsize=14, loc='upper right')
    plt.show()
    return

def L1_loss(y_true, y_pred):
    L1_loss = keras.mean(keras.abs(y_true - y_pred), axis=-1)
    return L1_loss
# L1 loss: MAE; L2 loss: MSE

def ssim_loss(y_true, y_pred):
    ssim_loss = 1 - (tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, max_val=1.0, filter_size=5)))
    return ssim_loss

def custom_loss(y_true, y_pred):
    L1_loss = keras.mean(keras.abs(y_true - y_pred), axis=-1)
    ssim_loss = 1 - (tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0, filter_size=5)))
    total_loss = 1.0 * L1_loss + 5 * ssim_loss
    return total_loss


# Compile the model
reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='auto', factor=0.2, patience=6, min_lr=0.00000001)
optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)  # SGD
metrics = ['mae', 'mse']

# model = build_3DUNet_model(img_height=128, img_width=128, img_depth=128, img_channels=3)
# model = build_dilated_3DUNet_model(img_height=128, img_width=128, img_depth=128, img_channels=1)
# model = build_res_3DUNet_model(img_height=128, img_width=128, img_depth=128, img_channels=1)
model = build_res_dilated_3DUNet_model(img_height=128, img_width=128, img_depth=128, img_channels=1)

model.compile(optimizer=optimizer, loss=custom_loss, metrics=metrics)
print(model.summary())
print(model.input_shape)
print(model.output_shape)

# Hyperparameters
epochs = 15
steps_per_epoch = len(train_img_t1ce_list) // batch_size
val_steps_per_epoch = len(val_img_t1ce_list) // batch_size

# Callbacks
checkpoint_filepath = 'saved_model/MR_CE_Synth_best_model.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5'
callbacks = [
    EarlyStopping(patience=50, monitor='val_loss', restore_best_weights=False, verbose=1),
    ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True),
    CSVLogger('MR_CE_Synth_3D_logs.csv', separator=',')]
# ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10, min_lr=1e-7, verbose=1)


# Load the model for continue training
model = load_model('saved_model/MR_CE_Synth_best_model.epoch02-loss0.41.hdf5',
                   custom_objects={'custom_loss': custom_loss})
print(model.summary())
print(model.input_shape)
print(model.output_shape)

# import tensorflow as tf
# tf.compat.v1.debugging.set_log_device_placement(True)
# print(tf.config.list_physical_devices('GPU'))
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# Fit the model
start = time.time()
history = model.fit(train_img_datagen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    verbose=1,
                    validation_data=val_img_datagen,
                    validation_steps=val_steps_per_epoch,
                    callbacks=[callbacks])

finish = time.time()
print('total exec. time (h)): ', (finish - start) / 3600.)
print('Training has been finished successfully')

# Save the trained model
model.save('saved_model/MR_CE_Synth_3D.hdf5')

# Plot the Learning Curve
filepath = 'MR_CE_Synth_3D_logs.csv'
plot_learning_curve(filepath)


###############################################################################
################################# THE END #####################################
###############################################################################
