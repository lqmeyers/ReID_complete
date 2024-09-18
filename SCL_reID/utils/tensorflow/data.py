import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from functools import partial
import pandas as pd
from sklearn.utils import shuffle
import os
#from reid_code.my_bees_augmentation_func import *
#from temp_folder.my_bees_augmentation_func import *
from data_augmentation import *
from IPython.display import Image
from PIL import Image as Image2

import ast
import time
from datetime import datetime

###########################################################################
#
# FUNCTIONS FOR LOADING AND FORMATTING DATASETS
#
###########################################################################




###################################################################################################
# FUNCTION FOR LOADING IMAGES
#
# INPUTS
# 1) filepath: string, filename or path of image
# 2) norm_method: int, specifies data normalization method
#                      0: do nothing (pixel values in [0,255] range)
#                      1: divide by 225 (pixel values in [0,1] range)
#                      2: divide by 127.5 and subtract 1 (pixel values in [-1,1] range)
#                      3) divide by 255 and center using provided per channel means and variance
# 3) mean_list: (optional) float list, list of mean pixel values per channel (required when norm_method==3)
# 4) std_list: (optional) float list, list of std pixel values per channel (required when norm_method==3)
# 5) channels: int, number of channels
#
# OUTPUTS
# 1) img: image as TF tensor
#
@tf.function 
def load_image(file_path, norm_method=0, mean_list=None, std_list=None, channels=3):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=channels)
    if norm_method == 1:
        # map pixels to [0,1] range (equiv divide by 255)
        # turns img from integer values to floats in (0,1) range
        img = tf.image.convert_image_dtype(img, tf.float32)
    elif norm_method == 2:
        # map pixels to [-1,1] range (i.e., divide by 127.5 and subtract 1)
        img = tf.cast(img, dtype=tf.float32)
        img = tf.math.divide(img, 127.5)
        img = tf.math.subtract(img, 1.0)
    elif norm_method == 3:
        # divide by 255 then center using provided per channel stats (mean and std)
        #img = tf.cast(img, dtype=tf.float32)
        #img = tf.divide(img, 255.0)
        img/=255
        img = tf.math.subtract(img, mean_list)
        img = tf.math.divide(img, std_list)
        
    return img
###################################################################################################


###################################################################################################
# FUNTION FOR CROPPING IMAGES
#
# INPUTS
# 1) image: image as TF tensor
# 2) h_range: int list, specifies first and last height coordinate for cropping
# 3) w_range: int list, specifies first and last width coordinate for cropping
#
# OUTPUTS
# cropped image
#
def crop_image(image, h_range, w_range):
    return image[h_range[0]:h_range[1], w_range[0]:w_range[1], :]
###################################################################################################


###################################################################################################
# FUNCTION FOR SPLITTING DATASET INTO TRAIN AND VALIDATION
# MAKES SPLITS ALONG LABLES
#
# INPUTS
# 1) df: pandas dataframe, containing dataset
# 2) label_col: string, name of column containing labels
# 3) train_frac: float, percent of samples to be assigned to training set
#
# OUTPUTS
# 1) train_df: pandas dataframe, containing training examples
# 2) valid_df: pandas dataframe, containing validation examples
#
def train_valid_split_df(df, label_col, train_frac=0.8):
    # get list of labels
    labels = df[label_col].unique()
    # split by labels
    # get number of training labels
    train_num = int(len(labels)*train_frac)
    # permute sample indices
    rand_labels = np.random.permutation(labels)
    # choose training labels
    train_labels = rand_labels[:train_num]
    # split df into train/val
    train_df = df[df[label_col].isin(train_labels)]
    valid_df = df[~df[label_col].isin(train_labels)]
    return train_df, valid_df
###################################################################################################


###################################################################################################
# FUNCTION FOR SPLITTING DATASET INTO TRAIN AND VALIDATION
# MAKES SPLITS PER LABELS
#
# INPUTS
# 1) df: pandas dataframe
# 2) label_col: str, name of column containing label/ID
# 3) train_percent: float, percent of samples assigned to train set
#
# OUTPUTS
# 1) df_train: pandas dataframe
# 2) df_val: pandas dataframe
#
def train_validation_split(df, label_col, train_percent=0.8):
    train_index_list = []
    for label in df[label_col].unique():
        index_list = df[df[label_col]==label].index.to_list()
        index_list = list(np.random.permutation(index_list))
        train_num = int(len(index_list)*train_percent)
        train_index_list+= index_list[:train_num]
    df_train = df[df.index.isin(train_index_list)]
    df_val = df[~df.index.isin(train_index_list)]
    return df_train, df_val
###################################################################################################



###################################################################################################
# FUNCTION FOR PREPARING DATASET FOR TRIPLET LOSS FUNCTION
#
# INPUTS
# 1) df: pandas dataframe, containing dataset
# 2) label_col: string, name of column containing labels
# 3) fname_col: string, name of column containing filenames or paths
#
# OUTPUTS
# 1) tdf: pandas dataframe, contains only filename and label coloumns
#
def prepare_for_triplet_loss(df, label_col, fname_col):
    # first sort by label value
    sdf = df.sort_values(label_col)
    # then extract labels and filenames from df
    labels = sdf[label_col].values
    filename = sdf[fname_col].values
    # then, make sure dataset has even number of samples
    # given remainder of function, wouldn't it make more sense to ensure each class has an
    # even number of samples?
    if labels.shape[0] % 2:
        labels = labels[1:]
        filename = filename[1:]
        
    # reshape lists into shape (K, 2) for some value K
    # presumably every row [i,:] will contain 2 samples with the same label (assuming even number of samples per label)
    pair_labels = labels.reshape((-1, 2))
    pair_filename = filename.reshape((-1, 2))
    # now permute the row indices
    ridx = np.random.permutation(pair_labels.shape[0])
    # rearrange lists by permuted row indices and flatten back to 1-D arrays
    labels = pair_labels[ridx].ravel()
    filename = pair_filename[ridx].ravel()
    # return as df
    tdf = pd.DataFrame({"filename":filename, "label":labels})
    return tdf
###################################################################################################


###################################################################################################
# FUNCTION FOR EXTRACTING FILENAMES AND LABELS
#
# INPUTS
# 1) df: pandas dataframe, contains dataset
# 2) label_col: string, name of column containing labels
# 3) fname_col: string, name of column containing filenames or paths
#
# OUTPUTS
# 1) file_path: string list, list of filenames or paths
# 2) labels: list, list of labels
#
def extract_filenames_and_labels(df, label_col, fname_col):
    file_path = list()
    labels = list()
    # get unique label list
    ids_list = list(df[label_col].unique())
    for i, row in df.iterrows():
        filename = row[fname_col]
        # rename label by index in unique label list
        y = ids_list.index(row[label_col])
        file_path.append(filename)
        labels.append(y)
    return file_path, labels
###################################################################################################





###################################################################################################
# ADD PARAMS FOR LAST FEW AUG METHODS
# FUNCTION THAT APPLIES AUGMENTATION TECHNIQUES TO IMAGES
#
# INPUTS
# 1) images: TF dataset, contains list of images
# 2) data_config: dictionary, contains necessary arguments for augmenting data, including the following
#                 required: 'aug_p', 'aug_methods', list of applicable augmentations, which are 'r_rotate, 'g_blur', 'c_jitter', 'c_drop', 'r_erase', 'r_sat',
#                                'r_bright', 'r_contrast', 'occlusion', 'add_color'
#                 optional (as needed): 'gblur_kernel', 'gblur_sigmin', 'gblur_sigmax', 'jitter_s', 'erase_sh', 'erase_r1',
#                                       'erase_method', 'color_coeff', 'color_N', 'sat_lower', 'sat_upper', 'bright_delta',
#                                       'cont_lower', 'cont_upper', 'occlude_h_range', 'occlude_w_range', 'occlude_val'
# 3) num_parallel_calls: int, used by some tensorflow functions, specifies "the number of batches to compute asynchronously in parallel" (Tensorflow)
#
# OUTPUTS
# 1) images: augmented images dataset
#
def apply_augmentations(images, data_config, num_parallel_calls=10):

    
    if 'r_translate' in data_config['aug_methods']:
        images = images.map(lambda x: random_translation(x, p=data_config['aug_p'], height_range=data_config['translate_hrange'], 
                                                         width_range=data_config['translate_wrange']), num_parallel_calls=num_parallel_calls)
    # random rotation
    if 'r_rotate' in data_config['aug_methods']:
        #images = images.map(random_rotation, num_parallel_calls=num_parallel_calls)
        images = images.map(lambda x: random_rotation(x, p=data_config['aug_p'], minval=data_config['rotate_min'], maxval=data_config['rotate_max']), 
                            num_parallel_calls=num_parallel_calls)
    # gaussian blur
    if 'g_blur' in data_config['aug_methods']:
        images = images.map(lambda x: gaussian_blur(x, data_config['aug_p'], data_config['gblur_kernel'], data_config['gblur_sigmin'], data_config['gblur_sigmax']), 
                            num_parallel_calls=num_parallel_calls)
    # color jitter
    if 'c_jitter' in data_config['aug_methods']:
        images = images.map(lambda x: color_jitter(x, data_config['aug_p'], data_config['jitter_s']), num_parallel_calls=num_parallel_calls)
    # color drop
    if 'c_drop' in data_config['aug_methods']:
        images = images.map(lambda x: color_drop(x, data_config['aug_p']), num_parallel_calls=num_parallel_calls)
    # random erase
    if 'r_erase' in data_config['aug_methods']:
        images = images.map(lambda x: random_erasing(x, data_config['aug_p'], data_config['erase_sl'], data_config['erase_sh'], data_config['erase_r1'], 
                                                     data_config['erase_method']), num_parallel_calls=num_parallel_calls)
    # random saturation
    if 'r_sat' in data_config['aug_methods']:
        images = images.map(lambda x: random_saturation(x, data_config['aug_p'], data_config['sat_lower'], data_config['sat_upper']), 
                            num_parallel_calls=num_parallel_calls)
    # random brightness
    if 'r_bright' in data_config['aug_methods']:
        images = images.map(lambda x: random_brightness(x, data_config['aug_p'], data_config['bright_delta']), num_parallel_calls=num_parallel_calls)
    # random contrast
    if 'r_contrast' in data_config['aug_methods']:
        images = images.map(lambda x: random_contrast(x, data_config['aug_p'], data_config['cont_lower'], data_config['cont_upper']), 
                            num_parallel_calls=num_parallel_calls)
    # random occlusion
    if 'occlusion' in data_config['aug_methods']:
        images = images.map(lambda x: occlude_image(x, data_config['occlude_h_range'], data_config['occlude_w_range'], data_config['aug_p']), 
                            num_parallel_calls=num_parallel_calls)
    # add color mask
    if 'add_color' in data_config['aug_methods']: 
        images = images.map(lambda x: add_color_mask(x, data_config['aug_p'], data_config['color_size_mean'], data_config['color_size_std'], 
                                                     data_config['color_H1_mean'], data_config['color_H1_std'], data_config['color_W1_mean'], 
                                                     data_config['color_W1_std']), num_parallel_calls=num_parallel_calls)
    
    return images
###################################################################################################


###################################################################################################
# FUNCTION FOR CREATING TF DATASET FOR UCL MODEL
# 
# INPUTS
# 1) df: pandas dataframe, dataframe containing containing data (images, labels)
# 2) data_config: dictionary, contains necessary arguments for loading data
#                 required: 'fname_col', 'label_col', 'cropped' 'input_size', 'augmentation'
#                 optional (as needed): 'h_range', 'w_range', 'aug_methods', 'mean', 'std'
#                 See apply_augmentations() for more required info
# 5) num_parallel_calls: int, used by some tensorflow functions, specifies "the number of batches to compute asynchronously in parallel" (Tensorflow)
#
# OUTPUTS
# 1) dataset: a tensorflow dataaset
#
def load_tf_pair_dataset(df, data_config, num_parallel_calls=10):
    x1_path_list = []
    label_list = []
    counter = 0
    for fname, label in zip(df[data_config['fname_col']].values, df[data_config['label_col']]):
        x1_path_list.append(fname)
        label_list.append(counter)
        counter+=1
        
    x1_path_list = np.array(x1_path_list)
    label_list = np.array(label_list)
    assert x1_path_list.shape[0] == label_list.shape[0]
    # shuffle
    index_list = np.arange(x1_path_list.shape[0])
    np.random.shuffle(index_list)
    x1_path_list = x1_path_list[index_list]
    x1_path_list = x1_path_list.copy()
    x2_path_list = x1_path_list.copy()
    label_list = label_list[index_list]
    
    x1_path_list = tf.data.Dataset.from_tensor_slices(x1_path_list)
    x2_path_list = tf.data.Dataset.from_tensor_slices(x2_path_list)
    label_list = tf.data.Dataset.from_tensor_slices(label_list)
    
    #x1_images = x1_path_list.map(lambda x: load_image(x, backbone, finetune), num_parallel_calls=num_parallel_calls)
    x1_images = x1_path_list.map(lambda x: load_image(x, data_config['norm_method'], data_config['mean'], data_config['std'], data_config['input_size'][-1]), num_parallel_calls=num_parallel_calls)
    if data_config['cropped']:
        x1_images = x1_images.map(lambda x: crop_image(x, data_config['h_range'], data_config['w_range']), num_parallel_calls=num_parallel_calls)
    # resize image if required
    sample = next(iter(x1_images.batch(1)))
    if sample.shape[1:-1] != data_config['input_size'][:2]:
        x1_images = x1_images.map(lambda x: tf.image.resize(x, data_config['input_size'][:2]), num_parallel_calls=num_parallel_calls)
        
    # if simCLR, augment both x1 and x2
    if data_config['simCLR']: 
        x1_images = apply_augmentations(x1_images, data_config, num_parallel_calls)
    
    x2_images = x2_path_list.map(lambda x: load_image(x, data_config['norm_method'], data_config['mean'], data_config['std'], data_config['input_size'][-1]), num_parallel_calls=num_parallel_calls)
    if data_config['cropped']:
        x2_images = x2_images.map(lambda x: crop_image(x, data_config['h_range'], data_config['w_range']), num_parallel_calls=num_parallel_calls)
    # resize image if required
    sample = next(iter(x2_images.batch(1)))
    if sample.shape[1:-1] != data_config['input_size'][:2]:
        x2_images = x2_images.map(lambda x: tf.image.resize(x, data_config['input_size'][:2]), num_parallel_calls=num_parallel_calls)
    # augment x2_images
    x2_images = apply_augmentations(x2_images, data_config, num_parallel_calls)
    
    dataset = tf.data.Dataset.zip((x1_images, x2_images, label_list))
    return dataset
###################################################################################################


###################################################################################################
# FUNCTION FOR CONSTRUCTING TF DATASET FOR SCL MODEL
# 
# INPUTS
# 1) df: pabdas dataframe, dataframe containing containing data (images, labels)
# 2) data_config: dictionary, contains necessary arguments for loading data
#                 required: 'cropped', 'rescale_factor', 'image_size', 'augmentation'
#                 optional (as needed): 'h_range', 'w_range', 'aug_methods'
#                 If using augmentation, see apply_augmentations() for more required info
# 3) label_col: string, name of column containing labels
# 4) fname_col: string, name of column containing filenames or paths
# 5) backbone: string, specifies backbone of model to be trained with data
# 6) finetune: bool, whether model to be trained is finetuned
# 7) mean_list: (optional) float list, list of mean pixel values per channel (for finetuning, if using mean centering with pre-trained dataset stats)
# 8) std_list: (optional) float list, list of std pixel values per channel (for finetuning, if using mean centering with pre-trained dataset stats)
# 9) validation: bool, whether dataset is validation set
# 10) num_parallel_calls: int, used by some tensorflow functions, specifies "the number of batches to compute asynchronously in parallel" (Tensorflow)
#
# OUTPUTS
# 1) dataset: a tensorflow dataset
#
def load_tf_dataset(df, data_config, label_col, fname_col, validation=False, num_parallel_calls=10):
    
    # get lists of filenames and labels
    filenames, labels = extract_filenames_and_labels(df, label_col, fname_col)
    # make lists into TensorSliceDataset
    filenames = tf.data.Dataset.from_tensor_slices(filenames)
    labels = tf.data.Dataset.from_tensor_slices(labels)
    # then make into MapDataset               
    images = filenames.map(lambda x: load_image(x, data_config['norm_method'], data_config['mean'], data_config['std'], data_config['input_size'][-1]), num_parallel_calls=num_parallel_calls)
    # crop images if specified
    if data_config['cropped'] == True and data_config['crop_before_aug'] == True:
        images = images.map(lambda x: crop_image(x, data_config['h_range'], data_config['w_range']), num_parallel_calls=num_parallel_calls)
    # add data augmentation if specified
    if data_config['augmentation'] == True and validation == False:
        images = apply_augmentations(images, data_config, num_parallel_calls)
    # crop images if specified
    if data_config['cropped'] == True and data_config['crop_before_aug']==False:
        images = images.map(lambda x: crop_image(x, data_config['h_range'], data_config['w_range']), num_parallel_calls=num_parallel_calls)
    # resize image if necessary
    images = images.map(lambda x: tf.image.resize(x, data_config['input_size'][:2]), num_parallel_calls=num_parallel_calls)
    dataset = tf.data.Dataset.zip((images, labels))
    return dataset
###################################################################################################


###################################################################################################
# INTERMEDIATE FUNCTION FOR CONSTRUCTING TF DATASET, TRAIN AND VALIDATION, FOR SCL MODEL
#
# INPUTS
# 1) train_df: pandas dataframe, dataframe containing training data
# 2) valid_df: pandas dataframe, dataframe containing validation data
# 3) data_config: dictionary, contains necessary arguments for loading data, see load_tf_dataset() for required info
# 3) label_col: string, name of column containing labels
# 4) fname_col: string, name of column containing filenames or paths
# 5) shuffle: bool, whether to shuffle the data
#
# OUTPUTS
# 1) train_dataset: train set as tensorflow dataset
# 2) valid_dataset: valid set as tensorflow dataset
#
def load_dataset(train_df, valid_df, data_config, label_col, fname_col, shuffle=True):
    
    train_dataset = load_tf_dataset(train_df, data_config, label_col, fname_col)
    valid_dataset = load_tf_dataset(valid_df, data_config, label_col, fname_col, validation=True)
    # shuffling if needed
    if shuffle:
        train_dataset = train_dataset.shuffle(len(train_df))
        valid_dataset = valid_dataset.shuffle(len(valid_df))    
    return train_dataset, valid_dataset
###################################################################################################


###################################################################################################
# FUNCTION TO LOAD DATA FOR SCL MODEL
# 
# INPUTS
# 1) data_config: dictionary, contains necessary arguments for loading data, including 'train_fname', 'valid_fname', 'label_col', 'fname_col', 'train_frac'
#                 see load_dataset() for more required info
# 2) verbose: bool, whether to print out messages
#
# OUTPUTS
# 1) train_dataset: train set as tensorflow dataset
# 2) valid_dataset: valid set as tensorflow dataset
#
def load_data(data_config, verbose=False):
    
    if verbose:
        print('Printing data_config:')
        for key, value in data_config.items():
            print(f'{key}: {value}')
    # split train data into train/validation if necessary
    if data_config['valid_fname'] is None:
        df = pd.read_csv(data_config['train_fname'])
        train_df, valid_df = train_valid_split_df(df, data_config['label_col'], train_frac=data_config['train_frac'])
    # else load previously split train/valid
    else:
        train_df = pd.read_csv(data_config['train_fname'])
        valid_df = pd.read_csv(data_config['valid_fname'])
    
    # resulting dfs have columns ['filename', 'label']
    train_df = prepare_for_triplet_loss(train_df, data_config['label_col'], data_config['fname_col'])
    valid_df = prepare_for_triplet_loss(valid_df, data_config['label_col'], data_config['fname_col'])
    
    train_dataset, valid_dataset = load_dataset(train_df, valid_df, data_config, 'label', 'filename', shuffle=False)
    
    return train_dataset, valid_dataset
###################################################################################################


###################################################################################################
# FUNCTION FOLR LOADING DATA FOR UCL MODEL
#
# INPUTS
# 1) data_config: dictionary, contains necessary arguments for loading data, including 'train_fname', 'valid_fname', 'label_col', 'fname_col', 'train_frac';
#                 see load_tf_pair_dataset() for more required info
# 2) verbose: bool, whether to print out messages
#
# OUTPUTS
# 1) train_dataset: train set as tensorflow dataset
# 2) valid_dataset: valid set as tensorflow dataset
#
def load_data_v2(data_config, verbose=False):

    if verbose:
        print('Printing data_config:')
        for key, value in data_config.items():
            print(f'{key}: {value}')

    # split train data into train/validation if necessary
    if data_config['valid_fname'] is None:
        df = pd.read_csv(data_config['train_fname'])
        train_df, valid_df = train_valid_split_df(df, data_config['label_col'], train_frac=data_config['train_frac'])
    # else load previously split train/valid
    else:
        train_df = pd.read_csv(data_config['train_fname'])
        valid_df = pd.read_csv(data_config['valid_fname'])

    
    train_df = train_df.sample(train_df.shape[0])
    valid_df = valid_df.sample(valid_df.shape[0])
    
    train_dataset = load_tf_pair_dataset(train_df, data_config)
    valid_dataset = load_tf_pair_dataset(valid_df, data_config)
    # shuffle
    #train_dataset = train_dataset.shuffle(len(train_df))
    #valid_dataset = valid_dataset.shuffle(len(valid_df))

    return train_dataset, valid_dataset
###################################################################################################


###################################################################################################
# THIS SHOULD NOT BE NEEDED - ELIMINATE
#
# CREATED TO MANUALLY RUN THE UNCOMPENSATED EXPERIMENTS
# LATER, CHANGE DF SO THAT THIS FUNCTION IS NOT NEEDED
# 
# INPUTS
# 1) data_config: dictionary, contains necessary arguments for loading data, including 'train_fname', 'valid_fname', 'label_col', 'fname_col', 'train_frac'
#                 see load_dataset() for more required info
# 2) verbose: bool, whether to print out messages
#
# OUTPUTS
# 1) train_dataset: train set as tensorflow dataset
# 2) valid_dataset: valid set as tensorflow dataset
#
def load_data_v3(data_config, verbose=False):

    if verbose:
        print('Printing data_config:')
        for key, value in data_config.items():
            print(f'{key}: {value}')
    # split train data into train/validation if necessary
    if data_config['valid_fname'] is None:
        df = pd.read_csv(data_config['train_fname'])
        train_df, valid_df = train_valid_split_df(df, data_config['label_col'], train_frac=data_config['train_frac'])
    # else load previously split train/valid
    else:
        train_df = pd.read_csv(data_config['train_fname'])
        valid_df = pd.read_csv(data_config['valid_fname'])

    # resulting dfs have columns ['filename', 'label']
    train_df = prepare_for_triplet_loss(train_df, data_config['label_col'], data_config['fname_col'])
    valid_df = prepare_for_triplet_loss(valid_df, data_config['label_col'], data_config['fname_col'])

    train_dataset, valid_dataset = load_dataset(train_df, valid_df, data_config, 'label', 'filename', shuffle=False)

    return train_dataset, valid_dataset
###################################################################################################


###################################################################################################
# FUNCTION FOR CONSTRUCTING TF DATASET FOR MTL MODEL
#
# INPUTS
# 1) df: pandas dataframe, contains data samples
# 2) data_config: dictionary, contains necessary arguments for loading data
#                 required: 'label_col', 'fname_col', 'meta_fname', 'color_col', 'cropped', 'input_size', 
#                            'augmentation'
#                 optional (as needed): 'h_range', 'w_range', 'color_set'
#                 If using augmentation, see apply_augmentations() for more required info
# 3) validation: bool, specifies whether df contains validation set samples
# 4) num_parallel_calls: int, used by some tensorflow functions, specifies "the number of batches to compute asynchronously in parallel" (Tensorflow)
# 5) verbose: bool, whether to print out messages
#
# OUTPUTS
# 1) dataset: a TF dataset
#
def load_tf_dataset_MTL(df, data_config, validation=False, num_parallel_calls=10, verbose=False):
    
    if verbose:
        if validation:
            print('loading tf validation data...')

    # first, prepare for triplet loss
    # resulting dataframe has columns ('label', 'filename')
    df = prepare_for_triplet_loss(df, data_config['label_col'], data_config['fname_col'])
    # get color info
    df_meta = pd.read_csv(data_config['meta_fname'])
    id_list = df_meta[data_config['label_col']].values
    # get list of possible colors
    if data_config['color_set'] is None:
        # if not user provided, extract from metadata dataframe
        color_set = []
        for entry in df_meta[data_config['color_col']].values:
            temp = ast.literal_eval(entry)
            color_set+=temp
        color_set = np.unique(color_set)
    else:
        color_set = np.array(data_config['color_set'])
    if verbose:
        print(f'color_set: {color_set}')
    # map colors to number IDs
    color_map = {val:k for k, val in enumerate(color_set)}
    if verbose:
        print(f'color_map: {color_map}')
    # binary color encoding for samples
    # a sample can have more than one position set to 1 (>1 color present)
    colors = np.zeros((df.shape[0], len(color_set)))
    if verbose:
        print(f'colors dim: {colors.shape}')
    for k, label in enumerate(df['label'].values):
        # turn string in dataframe into list
        temp = ast.literal_eval(df_meta[df_meta[data_config['label_col']]==label][data_config['color_col']].values[0])
        for c in temp:
            # for each color c in list, set the corresponding position to 1
            index = color_map[c]
            colors[k][index] = 1
    # get sample filenames
    filenames = df['filename'].values
    # get list of unique IDs
    ids_list = list(df['label'].unique())
    labels = []
    for i, row in df.iterrows():
        # rename label by index in unique label list
        y = ids_list.index(row['label'])
        labels.append(y)
    # then, proceed as usual 
    filenames = tf.data.Dataset.from_tensor_slices(filenames)
    labels = tf.data.Dataset.from_tensor_slices(labels)
    colors = tf.data.Dataset.from_tensor_slices(colors)
    images = filenames.map(lambda x: load_image(x, data_config['norm_method'], data_config['mean'], data_config['std'], data_config['input_size'][-1]), num_parallel_calls=num_parallel_calls)
    # crop images if required
    if data_config['cropped']:
        images = images.map(lambda x: crop_image(x, data_config['h_range'], data_config['w_range']), num_parallel_calls=num_parallel_calls)
    # resize images if required
    sample = next(iter(images.batch(1)))
    if sample.shape[1:-1] != data_config['input_size'][:2]:
        images = images.map(lambda x: tf.image.resize(x, data_config['input_size'][:2]), num_parallel_calls=num_parallel_calls)
    # apply specified augmentations
    if data_config['augmentation'] == True and validation == False:
        images = apply_augmentations(images, data_config, num_parallel_calls)
    # first, zip targets together
    targets = tf.data.Dataset.zip((labels, colors))
    # then zip inputs with targets
    dataset = tf.data.Dataset.zip((images, targets))
    
    return dataset
###################################################################################################



###################################################################################################
# FUNCTION FOR LOADING DATASET FOR MTL MODEL
#
# INPUTS
# 1) data_config: dictionary, contains necessary arguments for loading data, including 'train_fname', 'valid_fname', 'label_col', 'train_frac';
#                 see get_data_for_MTL_ID_color() for required info
# 2) verbose: bool, whether to print out messages
#
# OUTPUTS
# 1) train_dataset: train set as tensorflow dataset
# 2) valid_dataset: valid set as tensorflow dataset
#
def load_data_MTL(data_config, verbose=False):

    # split train data into train/validation if necessary
    if data_config['valid_fname'] is None:
        df = pd.read_csv(data_config['train_fname'])
        train_df, valid_df = train_valid_split_df(df, data_config['label_col'], train_frac=data_config['train_frac'])
    # else load previously split train/valid
    else:
        train_df = pd.read_csv(data_config['train_fname'])
        valid_df = pd.read_csv(data_config['valid_fname'])
    
    train_dataset = load_tf_dataset_MTL(train_df, data_config, validation = False, verbose=verbose)
    valid_dataset = load_tf_dataset_MTL(valid_df, data_config, validation = True, verbose=verbose)
    
    return train_dataset, valid_dataset
###################################################################################################



###################################################################################################
# FUNCTION FOR GETTING COLOR MAP
#
# INPUTS
# 1) data_config: dictionary, contains necessary parameters to construct color map, including 'meta_fname'
#                             and 'color_set'
# 2) verbose: bool, whether to print out comments
#
# OUTPUTS
# 1) color_map: dictionary, with color as key and color label as value
#
def get_color_map(data_config, verbose=False):
    # get color info
    df_meta = pd.read_csv(data_config['meta_fname'])
    
    # get list of possible colors
    if data_config['color_set'] is None:
        # if not user provided, extract from metadata dataframe
        color_set = []
        for entry in df_meta[data_config['color_col']].values:
            temp = ast.literal_eval(entry)
            color_set+=temp
        color_set = np.unique(color_set)
    else:
        color_set = np.array(data_config['color_set'])
    if verbose:
        print(f'color_set: {color_set}')
    # map colors to number IDs
    color_map = {val:k for k, val in enumerate(color_set)}
    if verbose:
        print(f'color_map: {color_map}')
        
    return color_map

###################################################################################################




###################################################################################################
# FUNCTION FOR CONSTRUCTING TF DATASET FOR COLOR PREDICTION MODEL
#
# INPUTS
# 1) df: pandas dataframe, contains data samples
# 2) data_config: dictionary, contains necessary arguments for loading data
#                 required: 'label_col', 'fname_col', 'meta_fname', 'color_col', 'cropped', 'input_size', 
#                            'augmentation'
#                 optional (as needed): 'h_range', 'w_range', 'color_set'
#                 If using augmentation, see apply_augmentations() for more required info
# 3) validation: bool, specifies whether df contains validation set samples
# 4) num_parallel_calls: int, used by some tensorflow functions, specifies "the number of batches to compute asynchronously in parallel" (Tensorflow)
# 5) verbose: bool, whether to print out messages
#
# OUTPUTS
# 1) dataset: a TF dataset
#
def load_tf_dataset_color(df, data_config, color_map=None, validation=False, num_parallel_calls=10, verbose=False):
    
    if verbose:
        if validation:
            print('loading tf validation data...')
    
    # get map of color to color_ID
    if color_map is None:
        color_map = get_color_map(data_config, verbose)

    # get color info
    df_meta = pd.read_csv(data_config['meta_fname'])
    # binary color encoding for samples
    # a sample can have more than one position set to 1 (>1 color present)
    colors = np.zeros((df.shape[0], len(color_map)))
    if verbose:
        print(f'colors dim: {colors.shape}')
    for k, label in enumerate(df[data_config['label_col']].values):
        # turn string in dataframe into list
        temp = ast.literal_eval(df_meta[df_meta[data_config['label_col']]==label][data_config['color_col']].values[0])
        for c in temp:
            # for each color c in list, set the corresponding position to 1
            index = color_map[c]
            colors[k][index] = 1
    # get sample filenames
    filenames = df[data_config['fname_col']].values
    
    filenames = tf.data.Dataset.from_tensor_slices(filenames)
    #print(filenames)
    colors = tf.data.Dataset.from_tensor_slices(colors)
    #print(colors)
    images = filenames.map(lambda x: load_image(x, data_config['norm_method'], data_config['mean'], data_config['std'], data_config['input_size'][-1]), num_parallel_calls=num_parallel_calls)
    #print(images)
    # crop images if required
    if data_config['cropped']:
        if verbose:
            print(f'cropping images with h_range {data_config["h_range"]} and w_range {data_config["w_range"]}...')
        images = images.map(lambda x: crop_image(x, data_config['h_range'], data_config['w_range']), num_parallel_calls=num_parallel_calls)
    # resize images if required
    sample = next(iter(images.batch(1)))
    if sample.shape[1:-1] != data_config['input_size'][:2]:
        images = images.map(lambda x: tf.image.resize(x, data_config['input_size'][:2]), num_parallel_calls=num_parallel_calls)
    # apply specified augmentations
    if data_config['augmentation'] == True and validation == False:
        if verbose:
            print('Augmenting images...')
        images = apply_augmentations(images, data_config, num_parallel_calls)
    dataset = tf.data.Dataset.zip((images, colors))
    
    return dataset, color_map
###################################################################################################



###################################################################################################
# FUNCTION FOR LOADING DATASET FOR COLOR PREDICTION MODEL
#
# INPUTS
# 1) data_config: dictionary, contains necessary arguments for loading data, including 'train_fname', 'valid_fname', 'label_col', 'train_frac';
#                 see get_data_for_MTL_ID_color() for required info
# 2) verbose: bool, whether to print out messages
#
# OUTPUTS
# 1) train_dataset: train set as tensorflow dataset
# 2) valid_dataset: valid set as tensorflow dataset
#
def load_data_color(data_config, verbose=False):

    # split train data into train/validation if necessary
    if data_config['valid_fname'] is None:
        df = pd.read_csv(data_config['train_fname'])
        train_df, valid_df = train_valid_split_df(df, data_config['label_col'], train_frac=data_config['train_frac'])
    # else load previously split train/valid
    else:
        train_df = pd.read_csv(data_config['train_fname'])
        valid_df = pd.read_csv(data_config['valid_fname'])
    
    train_dataset, color_map = load_tf_dataset_color(train_df, data_config, validation = False, verbose=verbose)
    valid_dataset, color_map = load_tf_dataset_color(valid_df, data_config, color_map, validation = True, verbose=verbose)
    
    return train_dataset, valid_dataset, color_map
###################################################################################################


# FUNCTION TO FORMAT DATA FOR COLOR CLASSIFIER
def load_tf_classifier_dataset(df, data_config, label_col, fname_col, validation=False, num_parallel_calls=10):
    
    # get lists of filenames and labels
    filenames = df[data_config['fname_col']].values
    labels = df[data_config['label_col']].values
    unique_labels = np.unique(labels)
    outputs = np.zeros((labels.shape[0], len(unique_labels)))
    for k, label in enumerate(labels):
        outputs[k][label] = 1
    # make lists into TensorSliceDataset
    filenames = tf.data.Dataset.from_tensor_slices(filenames)
    outputs = tf.data.Dataset.from_tensor_slices(outputs)
    # then make into MapDataset               
    images = filenames.map(lambda x: load_image(x, data_config['norm_method'], data_config['mean'], data_config['std'], data_config['input_size'][-1]), num_parallel_calls=num_parallel_calls)
    # crop images if specified
    if data_config['cropped'] == True and data_config['crop_before_aug'] == True:
        images = images.map(lambda x: crop_image(x, data_config['h_range'], data_config['w_range']), num_parallel_calls=num_parallel_calls)
    # add data augmentation if specified
    if data_config['augmentation'] == True and validation == False:
        images = apply_augmentations(images, data_config, num_parallel_calls)
    # crop images if specified
    if data_config['cropped'] == True and data_config['crop_before_aug']==False:
        images = images.map(lambda x: crop_image(x, data_config['h_range'], data_config['w_range']), num_parallel_calls=num_parallel_calls)
    # resize image if necessary
    images = images.map(lambda x: tf.image.resize(x, data_config['input_size'][:2]), num_parallel_calls=num_parallel_calls)
    dataset = tf.data.Dataset.zip((images, outputs))
    return dataset

def load_classifier_data(data_config, verbose=False):
    if verbose:
        print('Printing data_config:')
        for key, value in data_config.items():
            print(f'{key}: {value}')
    # split train data into train/validation if necessary
    if data_config['valid_fname'] is None:
        df = pd.read_csv(data_config['train_fname'])
        train_df, valid_df = train_valid_split_df(df, data_config['label_col'], train_frac=data_config['train_frac'])
    # else load previously split train/valid
    else:
        train_df = pd.read_csv(data_config['train_fname'])
        valid_df = pd.read_csv(data_config['valid_fname'])
    train_dataset = load_tf_classifier_dataset(train_df, data_config, data_config['label_col'], data_config['fname_col'])
    valid_dataset = load_tf_classifier_dataset(valid_df, data_config, data_config['label_col'], data_config['fname_col'], validation=True)
    # shuffling if needed
    if shuffle:
        train_dataset = train_dataset.shuffle(len(train_df))
        valid_dataset = valid_dataset.shuffle(len(valid_df))    
    return train_dataset, valid_dataset



# FUNCTIONS FOR TRACK MODEL


def prepare_for_triplet_loss_track(df, track_len=4,  repeats=10, label="label"):
    pairs = list()
    pair_labels = list()

    for i in range(repeats):

        ids = df[label].unique()
        shuffle(ids)

        A_df = df.groupby(label).sample(track_len, replace=True)
        A_df = A_df.set_index(label).loc[ids].reset_index()

        B_df = df.groupby(label).sample(track_len, replace=True)
        B_df = B_df.set_index(label).loc[ids].reset_index()

        A = A_df.filename.values
        B = B_df.filename.values
        A_label = A_df[label].values
        B_label = B_df[label].values

        pdf = np.hstack((A.reshape(-1, 1, track_len), B.reshape(-1, 1, track_len)))
        labels = np.dstack((A_label, B_label))
        
        pairs.append(pdf)
        pair_labels.append(labels)

    pair_df = np.vstack(pairs)
    pair_labels = np.vstack(pair_labels)
    df = pd.DataFrame({"filename": pair_df.ravel(), "label": pair_labels.ravel()})
    return df



def load_tf_track_dataset(df, track_len=5, rescale_factor=1, image_augmentation=False, augmentation=False, censored=True, label_column="track_tag_id"):
    
    filenames, labels = extract_filenames_and_labels(df, censored=censored, label_column=label_column)
#     classes = len(df.track_tag_id.unique())
        
    filenames = tf.data.Dataset.from_tensor_slices(filenames)
    labels = tf.data.Dataset.from_tensor_slices(labels[::track_len])
    
    images = filenames.map(load_image, num_parallel_calls=10)
    images = images.map(get_abdomen, num_parallel_calls=10)
    images = apply_rescale(images, rescale_factor=rescale_factor, image_size=(224, 224))
    
    if image_augmentation:
        images = images.map(gaussian_blur, num_parallel_calls=10)
        images = images.map(color_jitter, num_parallel_calls=10)
        images = images.map(color_drop, num_parallel_calls=10)
        images = images.map(random_erasing, num_parallel_calls=10)
        
    track_images = images.batch(track_len)
    
    if augmentation:
        track_images = track_images.map(track_gaussian_blur, num_parallel_calls=10)
        track_images = track_images.map(track_color_jitter, num_parallel_calls=10)
        track_images = track_images.map(track_color_drop, num_parallel_calls=10)
        track_images = track_images.map(track_random_erasing, num_parallel_calls=10)
        
    dataset = tf.data.Dataset.zip((track_images, labels))
    return dataset


def get_track_dataset(dataset, augmentation=False, track_len=4):
    
    if dataset == "untagged":
        label_column = "label"
        dataset_filename = DATASET_FILENAMES[dataset]
        untagged_df = pd.read_csv(dataset_filename)
        train_df, valid_df = train_valid_split_df(untagged_df, train_frac=0.8)
        train_df = prepare_for_triplet_loss_track(train_df, track_len=track_len, label=label_column)
        valid_df = prepare_for_triplet_loss_track(valid_df, track_len=track_len, label=label_column)
        
    elif dataset == "tagged":
        label_column = "track_tag_id"
        train_csv, valid_csv = DATASET_FILENAMES[dataset]
        train_df = pd.read_csv(train_csv)
        train_df = prepare_for_triplet_loss_track(train_df, label=label_column, track_len=track_len)
        valid_df = pd.read_csv(valid_csv)
        valid_df = prepare_for_triplet_loss_track(valid_df, label=label_column, track_len=track_len)
        
    
    train_dataset, valid_dataset = load_dataset_track(train_df, valid_df, track_len=track_len, augmentation=augmentation,  label_column="label", shuffle=False)
    return train_dataset, valid_dataset

def load_dataset_track(train_df, valid_df, track_len=4, augmentation=False, label_column="label", shuffle=True):
    
    train_dataset = load_tf_track_dataset(train_df, rescale_factor=4, track_len=track_len, augmentation=augmentation, label_column=label_column)
    valid_dataset = load_tf_track_dataset(valid_df, rescale_factor=4, track_len=track_len, label_column=label_column)
    
    if shuffle:
        train_dataset = train_dataset.shuffle(len(train_df))
        valid_dataset = valid_dataset.shuffle(len(valid_df))
    
    return train_dataset, valid_dataset






###################################################################################################
# FUNCTION FOR LOADING EVALUATION DATASET
# 
# INPUTS
# 1) filenames: string list, list of filenames or paths containing images
# 2) data_config: dictionary, contains necessary arguments for loading data, including 'norm_method', 'mean', 'std', 'input_size', 'cropped',
#                 'h_range', 'w_range';
#                 see get_data_for_MTL_ID_color() for required info
# 3) num_parallel_calls: int, used by some tensorflow functions, specifies "the number of batches to compute asynchronously in parallel" (Tensorflow)
#
# OUTPUTS
# 1) images: tensorflow dataset containing just images
#
def filename2image(filenames,data_config, num_parallel_calls=10):
    
    filenames = tf.data.Dataset.from_tensor_slices(filenames)
    images = filenames.map(lambda x: load_image(x, data_config['norm_method'], data_config['mean'], data_config['std'], data_config['input_size'][-1]), num_parallel_calls=num_parallel_calls)
    if data_config['cropped']:
        images = images.map(lambda x: crop_image(x, data_config['h_range'], data_config['w_range']), num_parallel_calls=num_parallel_calls)
    #func = lambda x: tf.image.resize(x, data_config['input_size'][:2])
    #images = images.map(func, num_parallel_calls=10)
    images = images.map(lambda x: tf.image.resize(x, data_config['input_size'][:2]), num_parallel_calls=num_parallel_calls)
    return images
###################################################################################################

