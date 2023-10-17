import numpy as np
import pandas as pd
from PIL import Image as Image2
import json 

import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Resize
import torchvision.transforms as transforms
from transformers import ViTModel, ViTFeatureExtractor


###################################################################################################
#
# PYTORCH VERSION OF DATA CODE
#
###################################################################################################



###################################################################################################
# CLASS FOR SINGLE IMAGE INPUT
# 
class Flowerpatch(Dataset):
    def __init__(self, df, fname_col, label_col, image_size, split, aug_p = 0.3):
        super(Flowerpatch, self).__init__()
        self.df = df
        self.fname_col = fname_col # column containing file name or path
        self.label_col = label_col # column containing label/ID
        self.image_size = image_size # image size, for Resize transform
        self.split = split # specifies dataset split (i.e., train vs valid vs test vs ref vs query)
        self.aug_p = aug_p # prob to apply data augmentation methods
        self.transform = transforms.Compose([transforms.Resize(image_size),
                                             transforms.ToTensor(),
                                            ])
        augmentation_methods = transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(degrees=(0, (3/2)*np.pi)), 
                                                                  transforms.ColorJitter(brightness=0.5, contrast=0.5)]), p=aug_p)
        self.train_transform = transforms.Compose([augmentation_methods,
                                                    transforms.Resize(image_size),
                                                    transforms.ToTensor()]) # include here augmentation techniques
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df.iloc[idx][self.label_col]
        label = torch.tensor(label, dtype=torch.long)
        img_path = self.df.iloc[idx][self.fname_col]
        image = Image2.open(img_path)
        # add transforms with data augmentation if train set
        if self.split == 'train':
            image = self.train_transform(image)
        else:
            image = self.transform(image)
        return {'image':image, 'label':label}
###################################################################################################


###################################################################################################
# CLASS FOR COLOR DETECTOR BINARY COLOR LABELS
#
# Uses the mapfile.json to substitute color maps for color code number
class ColorMap(Dataset):
    def __init__(self, df, fname_col, label_col, image_size, split, mapfile, aug_p = 0.3):
        super(ColorMap, self).__init__()
        self.df = df
        self.fname_col = fname_col # column containing file name or path
        self.label_col = label_col # column containing label/ID
        self.image_size = image_size # image size, for Resize transform
        self.split = split # specifies dataset split (i.e., train vs valid vs test vs ref vs query)
        self.aug_p = aug_p # prob to apply data augmentation methods
        with open(mapfile,'r') as f:
            self.mapfile = json.load(f) # path to file that contains dictionary of colormap values to substitute. 
        self.transform = transforms.Compose([transforms.Resize(image_size),
                                             transforms.ToTensor(),
                                            ])
        augmentation_methods = transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(degrees=(0, (3/2)*np.pi)), 
                                                                  transforms.ColorJitter(brightness=0.5, contrast=0.5)]), p=aug_p)
        self.train_transform = transforms.Compose([augmentation_methods,
                                                    transforms.Resize(image_size),
                                                    transforms.ToTensor()]) # include here augmentation techniques
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df.iloc[idx][self.label_col]
        label = self.mapfile[str(label)]
        label = torch.tensor(label, dtype=torch.float32)
        img_path = self.df.iloc[idx][self.fname_col]
        image = Image2.open(img_path)
        #replace improper channel images with blanks
        if len(np.array(image).shape) != 3:
            image = Image2.fromarray(np.zeros((self.image_size[0],self.image_size[1],3),dtype=np.int64))
        # add transforms with data augmentation if train set
        if self.split == 'train':
            image = self.train_transform(image)
        else:
            image = self.transform(image)
        return {'image':image, 'label':label}
###################################################################################################

###################################################################################################
# CLASS FOR COLOR DETECTOR BINARY COLOR LABELS
#
# Uses the mapfile.json to substitute color maps for color code number
#flattens array that encodes informatation 
class ColorMap_w_Order(Dataset):
    def __init__(self, df, fname_col, label_col, image_size, split, mapfile, aug_p = 0.3):
        super(ColorMap_w_Order, self).__init__()
        self.df = df
        self.fname_col = fname_col # column containing file name or path
        self.label_col = label_col # column containing label/ID
        self.image_size = image_size # image size, for Resize transform
        self.split = split # specifies dataset split (i.e., train vs valid vs test vs ref vs query)
        self.aug_p = aug_p # prob to apply data augmentation methods
        with open(mapfile,'r') as f:
            self.mapfile = json.load(f) # path to file that contains dictionary of colormap values to substitute. 
        self.transform = transforms.Compose([transforms.Resize(image_size),
                                             transforms.ToTensor(),
                                            ])
        augmentation_methods = transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(degrees=(0, (3/2)*np.pi)), 
                                                                  transforms.ColorJitter(brightness=0.5, contrast=0.5)]), p=aug_p)
        self.train_transform = transforms.Compose([augmentation_methods,
                                                    transforms.Resize(image_size),
                                                    transforms.ToTensor()]) # include here augmentation techniques
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df.iloc[idx][self.label_col]
        label = self.mapfile[str(label)]
        label = np.array(label)
        label = label.flatten()
        label = torch.tensor(label, dtype=torch.float32)
        img_path = self.df.iloc[idx][self.fname_col]
        image = Image2.open(img_path)
        #replace improper channel images with blanks
        if len(np.array(image).shape) != 3:
            image = Image2.fromarray(np.zeros((self.image_size[0],self.image_size[1],3),dtype=np.int64))
        # add transforms with data augmentation if train set
        if self.split == 'train':
            image = self.train_transform(image)
        else:
            image = self.transform(image)
        return {'image':image, 'label':label}
###################################################################################################

####################################################################################################
# CLASS FOR MULTITASK LEARNING 
# RETURNS TWO LABELS 
# Currently COLOR CODE IS REID TARGET, NOT IDENTITY
class MultiTaskData(Dataset):
    def __init__(self, df, fname_col, label_col, image_size, split, mapfile, aug_p = 0.3):
        super(MultiTaskData, self).__init__()
        self.df = df
        self.fname_col = fname_col # column containing file name or path
        self.label_col = label_col # column containing label/ID
        self.image_size = image_size # image size, for Resize transform
        self.split = split # specifies dataset split (i.e., train vs valid vs test vs ref vs query)
        self.aug_p = aug_p # prob to apply data augmentation methods
        with open(mapfile,'r') as f:
            self.mapfile = json.load(f) # path to file that contains dictionary of colormap values to substitute. 
        self.transform = transforms.Compose([transforms.Resize(image_size),
                                             transforms.ToTensor(),
                                            ])
        augmentation_methods = transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(degrees=(0, (3/2)*np.pi)), 
                                                                  transforms.ColorJitter(brightness=0.5, contrast=0.5)]), p=aug_p)
        self.train_transform = transforms.Compose([augmentation_methods,
                                                    transforms.Resize(image_size),
                                                    transforms.ToTensor()]) # include here augmentation techniques
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df.iloc[idx][self.label_col]
        color_label = self.mapfile[str(label)]
        label = torch.tensor(label, dtype=torch.float32)
        color_label = torch.tensor(color_label, dtype=torch.float32)
        img_path = self.df.iloc[idx][self.fname_col]
        image = Image2.open(img_path)
        #replace improper channel images with blanks
        if len(np.array(image).shape) != 3:
            image = Image2.fromarray(np.zeros((self.image_size[0],self.image_size[1],3),dtype=np.int64))
        # add transforms with data augmentation if train set
        if self.split == 'train':
            image = self.train_transform(image)
        else:
            image = self.transform(image)
        return {'image':image, 'label':label,'color_label':color_label}
###################################################################################################
# CLASS FOR TRACK INPUTS
# TO BE USED WITH SWIN3D MODEL
#
class TrackData(Dataset):
    def __init__(self, track_df, image_df, fname_col, label_col, track_col, track_len, image_size):
        super(TrackData, self).__init__()
        self.track_df = track_df
        self.image_df = image_df
        self.fname_col = fname_col # column containing file name or path
        self.label_col = label_col # column containing label/ID
        self.track_col = track_col # column containing track ID
        self.track_len = track_len # number of images per track
        self.image_size = image_size # image size, for Resize transform
        # transform for Swin3D model
        #self.transform = torchvision.models.video.Swin3D_S_Weights.KINETICS400_V1.transforms()
        #self.transform = torchvision.models.video.Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1.transforms()
        # transforms as swin3d would apply, without the normalization
        self.transform = transforms.Compose([transforms.Resize(self.image_size),
                                             transforms.ToTensor(),
                                            ])

    def __len__(self):
        # length determined by number of unique tracks
        return len(self.track_df)

    def __getitem__(self, idx):
        # get track ID and corresponding label
        track = self.track_df.iloc[idx][self.track_col]
        label = self.track_df.iloc[idx][self.label_col]
        # get filepaths for images of track
        img_path_list = self.image_df[self.image_df[self.track_col]==track][self.fname_col].values
        # if track has more images, limit to the first track_len images
        if len(img_path_list) > self.track_len:
            img_path_list = img_path_list[:self.track_len]
        img_list = []
        #for img_path in img_path_list:
        #    img = Image2.open(img_path)
        #    img = np.array(img)
        #    img_list.append(img)
        # make channels "first"; results in (track_len, channels, height, width)
        #image = np.swapaxes(np.stack(img_list), -1, 1)
        #image = torch.Tensor(image)
        #image = self.transform(image)
        for img_path in img_path_list:
            img = Image2.open(img_path)
            img = self.transform(img)
            img_list.append(img)
        image = torch.stack(img_list)
        image = torch.swapaxes(image, 0, 1)
        label = torch.tensor(label, dtype=torch.long)
        return {'image':image, 'label':label}
###################################################################################################


###################################################################################################
# FUNCTION TO GET DATASET
# 
def get_dataset(data_config, split):
    df = pd.read_csv(data_config['datafiles'][split])
    dataset = Flowerpatch(df, data_config['fname_col'], data_config['label_col'], data_config['input_size'], split, data_config['aug_p'])
    # only shuffle if train
    if split == 'train':
        dataloader = DataLoader(dataset, batch_size=data_config['batch_size'], shuffle=True)
    else:
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    return dataloader
###################################################################################################


###################################################################################################
#  FUNCTION TO GET GALLERIES
# 
def get_galleries(data_config):
    dataframe_file = data_config['datafiles']['gallery']
    df = pd.read_csv(dataframe_file)
    dataset = Flowerpatch(df, data_config['fname_col'], 'image_id', data_config['input_size'], 'gallery')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    return dataloader
###################################################################################################

def get_track_dataset(data_config, split):
    track_file = data_config['datafiles'][split]['track_file']
    track_df = pd.read_csv(track_file)
    image_file = data_config['datafiles'][split]['image_file']
    image_df = pd.read_csv(image_file)
    dataset = TrackData(track_df, image_df, data_config['fname_col'], data_config['label_col'], 
                        data_config['track_col'], data_config['track_len'], data_config['input_size'])
    # only shuffle if train
    if split == 'train':
        dataloader = DataLoader(dataset, batch_size=data_config['batch_size'], shuffle=True)
    else:
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    return dataloader



