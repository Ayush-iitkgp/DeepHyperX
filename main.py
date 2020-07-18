# -*- coding: utf-8 -*-
"""
DEEP LEARNING FOR HYPERSPECTRAL DATA.

This script allows the user to run several deep models (and SVM baselines)
against various hyperspectral datasets. It is designed to quickly benchmark
state-of-the-art CNNs on various public hyperspectral datasets.

This code is released under the GPLv3 license for non-commercial and research
purposes only.
For commercial use, please contact the authors.
"""
# Python 2/3 compatiblity
from __future__ import print_function
from __future__ import division

# Torch
import torch
import torch.utils.data as data
from torchsummary import summary

# Numpy, scipy, scikit-image, spectral
import numpy as np
import sklearn.svm
import sklearn.model_selection
from skimage import io
# Visualization
import seaborn as sns
import visdom
from PIL import Image
from scipy.special import softmax


import os
from utils import metrics, convert_to_color_, convert_from_color_,\
    display_dataset, display_predictions, explore_spectrums, plot_spectrums,\
    sample_gt, build_dataset, show_results, compute_imf_weights, get_device
from datasets import get_dataset, HyperX, open_file, DATASETS_CONFIG
from models import get_model, train, test, save_model

import argparse

dataset_names = [v['name'] if 'name' in v.keys() else k for k, v in DATASETS_CONFIG.items()]

# Argument parser for CLI interaction
parser = argparse.ArgumentParser(description="Run deep learning experiments on"
                                             " various hyperspectral datasets")
parser.add_argument('--dataset', type=str, default=None, choices=dataset_names,
                    help="Dataset to use.")
parser.add_argument('--model', type=str, default=None,
                    help="Model to train. Available:\n"
                    "SVM (linear), "
                    "SVM_grid (grid search on linear, poly and RBF kernels), "
                    "baseline (fully connected NN), "
                    "hu (1D CNN), "
                    "hamida (3D CNN + 1D classifier), "
                    "lee (3D FCN), "
                    "chen (3D CNN), "
                    "li (3D CNN), "
                    "he (3D CNN), "
                    "luo (3D CNN), "
                    "sharma (2D CNN), "
                    "boulch (1D semi-supervised CNN), "
                    "liu (3D semi-supervised CNN), "
                    "mou (1D RNN)")
parser.add_argument('--folder', type=str, help="Folder where to store the "
                    "datasets (defaults to the current working directory).",
                    default="./Datasets/")
parser.add_argument('--cuda', type=int, default=-1,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")
parser.add_argument('--runs', type=int, default=1, help="Number of runs (default: 1)")
parser.add_argument('--restore', type=str, default=None,
                    help="Weights to use for initialization, e.g. a checkpoint")

# Dataset options
group_dataset = parser.add_argument_group('Dataset')
group_dataset.add_argument('--training_sample', type=float, default=10,
                    help="Percentage of samples to use for training (default: 10%%)")
group_dataset.add_argument('--sampling_mode', type=str, help="Sampling mode"
                    " (random sampling or disjoint, default: random)",
                    default='random')
group_dataset.add_argument('--train_set', type=str, default=None,
                    help="Path to the train ground truth (optional, this "
                    "supersedes the --sampling_mode option)")
group_dataset.add_argument('--test_set', type=str, default=None,
                    help="Path to the test set (optional, by default "
                    "the test_set is the entire ground truth minus the training)")
# Training options
group_train = parser.add_argument_group('Training')
group_train.add_argument('--epoch', type=int, help="Training epochs (optional, if"
                    " absent will be set by the model)")
group_train.add_argument('--patch_size', type=int,
                    help="Size of the spatial neighbourhood (optional, if "
                    "absent will be set by the model)")
group_train.add_argument('--lr', type=float,
                    help="Learning rate, set by the model if not specified.")
group_train.add_argument('--class_balancing', action='store_true',
                    help="Inverse median frequency class balancing (default = False)")
group_train.add_argument('--batch_size', type=int,
                    help="Batch size (optional, if absent will be set by the model")
group_train.add_argument('--test_stride', type=int, default=1,
                     help="Sliding window step stride during inference (default = 1)")
# Data augmentation parameters
group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true',
                    help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true',
                    help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true',
                    help="Random mixes between spectra")

parser.add_argument('--with_exploration', action='store_true',
                    help="See data exploration visualization")
parser.add_argument('--download', type=str, default=None, nargs='+',
                    choices=dataset_names,
                    help="Download the specified datasets and quits.")



args = parser.parse_args()

CUDA_DEVICE = get_device(args.cuda)

# % of training samples
SAMPLE_PERCENTAGE = args.training_sample
# Data augmentation ?
FLIP_AUGMENTATION = args.flip_augmentation
RADIATION_AUGMENTATION = args.radiation_augmentation
MIXTURE_AUGMENTATION = args.mixture_augmentation
# Dataset name
DATASET = args.dataset
# Model name
MODEL = args.model
# Number of runs (for cross-validation)
N_RUNS = args.runs
# Spatial context size (number of neighbours in each spatial direction)
PATCH_SIZE = args.patch_size
# Add some visualization of the spectra ?
DATAVIZ = args.with_exploration
# Target folder to store/download/load the datasets
FOLDER = args.folder
# Number of epochs to run
EPOCH = args.epoch
# Sampling mode, e.g random sampling
SAMPLING_MODE = args.sampling_mode
# Pre-computed weights to restore
CHECKPOINT = args.restore
# Learning rate for the SGD
LEARNING_RATE = args.lr
# Automated class balancing
CLASS_BALANCING = True
# CLASS_BALANCING = args.class_balancing
# Training ground truth file
TRAIN_GT = args.train_set
# Testing ground truth file
TEST_GT = args.test_set
TEST_STRIDE = args.test_stride

# if args.download is not None and len(args.download) > 0:
#     for dataset in args.download:
#         get_dataset(dataset, target_folder=FOLDER)
#     quit()

# viz = visdom.Visdom(env=DATASET + ' ' + MODEL)
# if not viz.check_connection:
#     print("Visdom is not connected. Did you run 'python -m visdom.server' ?")


hyperparams = vars(args)
# Load the dataset
# img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET,
                                                            #    FOLDER)
def file_name_train(subject_id):
    mapname = '{}.png'.format('Prashant/Train_Data/Map/TumorMap' + str(subject_id))
    tiffname = '{}.tiff'.format('Prashant/Train_Data/ROI/rfImages' + str(subject_id))
    return mapname, tiffname

def file_name_test(subject_id):
    mapname = '{}.png'.format('Prashant/Test_Data/Map/TumorMap' + str(subject_id))
    tiffname = '{}.tiff'.format('Prashant/Test_Data/ROI/rfImages' + str(subject_id))
    return mapname, tiffname

LABEL_VALUES = [0, 1]
IGNORED_LABELS = []
RGB_BANDS = [] 
# Number of classes
N_CLASSES = len(LABEL_VALUES)
# Number of bands (last dimension of the image tensor)
N_BANDS = 55

def get_training_data(train_num):
    img = np.empty(shape=[0, 540, 55], dtype=np.float32)
    train_gt =  np.empty(shape=[0,540], dtype=np.float32)
    for num in train_num:
        map_image,tiff_image = file_name_train(num)
        img_k = Image.open(map_image)
        img_arr = np.array(img_k)
        img_arr = img_arr
        y_tmp = img_arr[:,:,1] == 255
        train_gt = np.append(train_gt, y_tmp, axis = 0)

        im = Image.open(tiff_image)
        h,w = np.array(im).shape
        tiffarray = np.zeros((h,w,im.n_frames))

        for i in range(im.n_frames):
            im.seek(i)
            tiffarray[:,:,i] = np.array(im)
    
        img = np.append(img, tiffarray, axis = 0)
    return img, train_gt

def get_testing_data(test_num):
    img_test = np.empty(shape=[0, 540, 55], dtype=np.float32)
    test_gt =  np.empty(shape=[0,540], dtype=np.float32)

    for num in test_num:
        map_image,tiff_image = file_name_test(num)
        img_k = Image.open(map_image)
        img_arr = np.array(img_k)
        img_arr = img_arr
        y_tmp = img_arr[:,:,1] == 255
        test_gt = np.append(test_gt, y_tmp, axis = 0)

        im = Image.open(tiff_image)
        h,w = np.array(im).shape
        tiffarray = np.zeros((h,w,im.n_frames))

        for i in range(im.n_frames):
            im.seek(i)
            tiffarray[:,:,i] = np.array(im)
        # tiffarray = tiffarray.
        img_test = np.append(img_test, tiffarray, axis = 0)
    return img_test, test_gt

# Parameters for the SVM grid search
SVM_GRID_PARAMS = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3],
                                       'C': [1, 10, 100, 1000]},
                   {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]},
                   {'kernel': ['poly'], 'degree': [3], 'gamma': [1e-1, 1e-2, 1e-3]}]

palette = None
if palette is None:
    # Generate color palette
    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", len(LABEL_VALUES) - 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))
invert_palette = {v: k for k, v in palette.items()}

def convert_to_color(x):
    return convert_to_color_(x, palette=palette)
def convert_from_color(x):
    return convert_from_color_(x, palette=invert_palette)


# Instantiate the experiment based on predefined networks
hyperparams.update({'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 'device': CUDA_DEVICE})
hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

train_num = [61, 62, 100, 101, 120, 121, 123, 138]
# train_num = [61, 62]
test_num = [76, 77]

results = []
N_RUNS = 1
img, train_gt = get_training_data(train_num)
if CLASS_BALANCING:
    weights = compute_imf_weights(train_gt, N_CLASSES, IGNORED_LABELS)
    print(weights.shape)
    print(weights.dtype)
    hyperparams['weights'] = torch.from_numpy(weights).float()

model, optimizer, loss, hyperparams = get_model(MODEL, **hyperparams)

if CHECKPOINT is not None:
    model.load_state_dict(torch.load(CHECKPOINT))

for run in range(N_RUNS):
    img, train_gt = get_training_data(train_num)
    train_gt, val_gt = sample_gt(train_gt, SAMPLE_PERCENTAGE, mode='random')
    # Generate the dataset
    train_dataset = HyperX(img, train_gt, **hyperparams)
    # print(train_dataset.data.shape)
    # print(train_dataset.data.dtype)
    # print(train_dataset.label.dtype)
    train_loader = data.DataLoader(train_dataset,
                                       batch_size=hyperparams['batch_size'],
                                       #pin_memory=hyperparams['device'],
                                       shuffle=True)
    val_dataset = HyperX(img, val_gt, **hyperparams)
    print(val_dataset.data.shape)
    val_loader = data.DataLoader(val_dataset,
                                     #pin_memory=hyperparams['device'],
                                     batch_size=hyperparams['batch_size'])

    print(hyperparams)
    print("Network :")
    with torch.no_grad():
        for input, _ in train_loader:
            break
        summary(model.to(hyperparams['device']), input.size()[1:])

    try:
        pass
        # train(model, optimizer, loss, train_loader, hyperparams['epoch'],
        #           scheduler=hyperparams['scheduler'], device=hyperparams['device'],
        #           supervision=hyperparams['supervision'], val_loader=val_loader)
    except KeyboardInterrupt:
        pass

if N_RUNS >= 1:
    img_test, test_gt = get_testing_data(test_num)
    probabilities = test(model, img_test.astype(np.double), hyperparams)
    prob = softmax(probabilities, axis = 2)
    
    # prediction = 
    prediction = np.argmax(prob, axis=-1)

    run_results = metrics(prediction, test_gt, ignored_labels=hyperparams['ignored_labels'], n_classes=N_CLASSES)
    print(run_results)

    mask = np.zeros(test_gt.shape, dtype='bool')
    for l in IGNORED_LABELS:
        mask[test_gt == l] = True
    prediction[mask] = 0

    color_prediction = convert_to_color(prediction)
    results.append(run_results)

    show_results(results, None, label_values=LABEL_VALUES, agregated=True)

