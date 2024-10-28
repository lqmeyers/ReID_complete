import numpy as np
import pandas as pd
from PIL import Image as Image2
import json 
import os 

import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Resize
import torchvision.transforms as transforms
from transformers import ViTModel, ViTFeatureExtractor



###################################################################################################
#
# PYTORCH DATASET DATALOADERS FOR MODEL TRAINING
#
###################################################################################################


##########################################################################################
# FUNCTION TO GET EMBEDDINGS AND LABELS FOR EVALUATING MODEL
def get_embeddings(model, dataloader, loss_fn, miner, device, feature_extractor=None):
    embeddings = []
    all_labels = []
    loss = 0.0
    with torch.no_grad():
        for k, batch in enumerate(dataloader):
            if feature_extractor is None:
                images = batch['image'].to(device)
            else:
                images = [transforms.functional.to_pil_image(x) for x in batch['image']]
                images = np.concatenate([feature_extractor(x)['pixel_values'] for x in images])
                images = torch.tensor(images, dtype=torch.float).to(device)
            labels = batch['label'].to(device)
            outputs = model(images)
            hard_pairs = miner(outputs, labels)
            loss += loss_fn(outputs, labels, hard_pairs).detach().cpu().numpy()
            embeddings.append(outputs.detach().cpu().numpy())
            all_labels += list(labels.detach().cpu().numpy())
    embeddings = np.vstack(embeddings)
    all_labels = np.array(all_labels)
    loss/=k
    return embeddings, all_labels, loss
##########################################################################################


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
        augmentation_methods = transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(degrees=(0, 2*np.pi)), 
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
        #make imag nparray for standardization 
        #image = np.array(image)
        return {'image':image, 'label':label}
###################################################################################################

class Flowerpatch_w_Track(Dataset):
    def __init__(self, df, fname_col, label_col, track_col, image_size, split, aug_p = 0.3):
        super(Flowerpatch_w_Track, self).__init__()
        self.df = df
        self.fname_col = fname_col # column containing file name or path
        self.label_col = label_col # column containing label/ID
        self.image_size = image_size # image size, for Resize transform
        self.track_col = track_col #column containing track information
      
        self.split = split # specifies dataset split (i.e., train vs valid vs test vs ref vs query)
        self.aug_p = aug_p # prob to apply data augmentation methods
        self.transform = transforms.Compose([transforms.Resize(image_size),
                                             transforms.ToTensor(),
                                            ])
        augmentation_methods = transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(degrees=(0, 2*np.pi)), 
                                                                  transforms.ColorJitter(brightness=0.5, contrast=0.5)]), p=aug_p)
        self.train_transform = transforms.Compose([augmentation_methods,
                                                    transforms.Resize(image_size),
                                                    transforms.ToTensor()]) # include here augmentation techniques
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        id_label = self.df.iloc[idx][self.label_col]
        id_label = torch.tensor(id_label, dtype=torch.long)
        track_label = self.df.iloc[idx][self.track_col]
        track_label = torch.tensor(track_label, dtype=torch.long)
        img_path = self.df.iloc[idx][self.fname_col]
        image = Image2.open(img_path)
        # add transforms with data augmentation if train set
        if self.split == 'train':
            image = self.train_transform(image)
        else:
            image = self.transform(image)
        return {'image':image, 'label':id_label,'track':track_label}
###################################################################################################




##################################################################################################
# Flowerpatch Dataset Class that also includes track labels, in order to precompute 
# embeddings but maintain track distinctions
###################################################################################################

class Flowerpatch_w_Track_and_Filter(Dataset):
    def __init__(self, df, fname_col, label_col, track_col, image_size, split, imgs_per_track=6, aug_p = 0.3):
        super(Flowerpatch_w_Track_and_Filter, self).__init__()
        self.df = df
        self.fname_col = fname_col # column containing file name or path
        self.label_col = label_col # column containing label/ID
        self.image_size = image_size # image size, for Resize transform
        self.track_col = track_col #column containing track information
        self.imgs_per_track = imgs_per_track
        self.split = split # specifies dataset split (i.e., train vs valid vs test vs ref vs query)
        self.aug_p = aug_p # prob to apply data augmentation methods
        self.transform = transforms.Compose([transforms.Resize(image_size),
                                             transforms.ToTensor(),
                                            ])
        augmentation_methods = transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(degrees=(0, 2*np.pi)), 
                                                                  transforms.ColorJitter(brightness=0.5, contrast=0.5)]), p=aug_p)
        self.train_transform = transforms.Compose([augmentation_methods,
                                                    transforms.Resize(image_size),
                                                    transforms.ToTensor()]) # include here augmentation techniques
        
        #print("Full:",len(self.df))
        #  filter df to only contain long enough tracks
        self.df = self.df.groupby([self.track_col,self.label_col]).filter(lambda x: len(x) > self.imgs_per_track)
        #print("With enough imgs per track:",len(self.df))
        #  remove id with only 1 track
        self.df = self.df.groupby(self.label_col).filter(lambda x: len(np.unique(x[self.track_col])) > 1)
        #print("With only ids with multiple filtered tracks:",len(self.df))
        self.df_len = len(df)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        id_label = self.df.iloc[idx][self.label_col]
        id_label = torch.tensor(id_label, dtype=torch.long)
        track_label = self.df.iloc[idx][self.track_col]
        track_label = torch.tensor(track_label, dtype=torch.long)
        img_path = self.df.iloc[idx][self.fname_col]
        image = Image2.open(img_path)
        # add transforms with data augmentation if train set
        if self.split == 'train':
            image = self.train_transform(image)
        else:
            image = self.transform(image)
        return {'image':image, 'label':id_label,'track':track_label}
###################################################################################################

#for filtering all tensor arrays need to convert np indices to tensor and initilize mat tens array
#with the same shape as array to be filtered, change values to be removed/kept and then filter
#tensor by tensor, may cause flattening problems... investigating

##################################################################################################
## Dataset class that uses precomputed tensor array
## returns tensor array of len imgs per track that contains a set of embeddings from same id and same track
##################################################################################################
class Flowerpatch_Embeddings_v2(Dataset):
    def __init__(self, emb_tens_arr, id_arr, track_arr,imgs_per_track=5):
        super(Flowerpatch_Embeddings_v2, self).__init__()
        self.emb_tens_arr = emb_tens_arr #tensor array of embeddings
        self.id_arr = id_arr # np array of labels
        self.track_arr = track_arr #np array of track ids
        self.imgs_per_track = imgs_per_track # number of images in each track to feed 
        self.df = self.build_df()
    
    def __len__(self):
        return len(self.emb_tens_arr)

    def build_df(self):
        #save sets of embeddings by index in dataframe
        df = pd.DataFrame({"ID":self.id_arr.flatten(),"track":self.track_arr.flatten()})
        df['Pairs_indices'] = df.apply(lambda row: self.sample_track_id_image(row['ID'], row['track']), axis=1)
        return df 

    def sample_track_id_image(self,id,track_id):
        #return the index of a randomly selected embedding 
        #with the same id and track_id
        idxs_same_id = np.where(self.id_arr == id)[0] #get indices of other samples of current id
        tracks_to_check = self.track_arr[idxs_same_id] #filter tracks to check within id
        idxs_same_track = np.where(tracks_to_check == track_id)[0] #get indicies that are same track and id
        if len(idxs_same_track) >= self.imgs_per_track -1:
            random_indices = np.random.choice(idxs_same_track, size=(self.imgs_per_track - 1), replace=False) #sample random indicies
        else:
            print(f"Warning: Not enough items in track {track_id} of id {id} with to sample {self.imgs_per_track} images.")
            print("Number of image embeddings to sample among the", len(idxs_same_track),"elements because size is",idxs_same_track.shape)
            random_indices = np.random.choice(idxs_same_track, size=(self.imgs_per_track - 1), replace=True) #sample random indicies
        return random_indices

    def __getitem__(self, idx):            
        anchor_indices = self.df.loc[idx, 'Pairs_indices']
        anchor_indices = torch.tensor(anchor_indices) #convert np idx array to tensor
        
        #Create a mask for selecting rows
        mask = torch.zeros_like(self.emb_tens_arr, dtype=torch.bool)
        mask[anchor_indices, :] = True

        # Select rows from the embedding tensor based on the mask
        anchor_embs = self.emb_tens_arr[mask]

        # Reshape the output to ensure the correct shape
        anchor_embs = anchor_embs.view(-1, self.emb_tens_arr.size(1))
               
        #Add original sample to others in track
        sample_emb = self.emb_tens_arr[idx].unsqueeze(0)
        anchor_embs = anchor_embs #might need to unsqueeze ? 

        anchor_embs = torch.cat((sample_emb,anchor_embs)) #append tracks together 
        id = torch.tensor(self.id_arr[idx])

        return {'track_embeddings':anchor_embs,'id':id}

############################# Function for sampling multi-instance samples 

## a pd method to be used with apply
def get_track_id_image(df_in, idx,imgs_per_track):
    #return the index of all potential samples
    #with the same id and track_id
    
    #find id and track from input idx
    id = df_in["ID"][idx]
    track_id = df_in["Track"][idx]
    
    #drop idx from df to not include in set to be sampled
    df_subset = df_in.drop(idx)
        
    rows_same_id = df_subset[df_subset["ID"] == id] #get rows of other samples of current id
    rows_same_track = rows_same_id[rows_same_id["Track"] == track_id] #get rows that are same track and id
    #print(f"Found {len(rows_same_track)} rows with same track and id")
       
    return rows_same_track.index.values


############################# Dataset Class to use precomputed embeddings for multiinstance #############

class Flowerpatch_Embeddings_from_df(Dataset):
    def __init__(self, emb_df,imgs_per_track=5):
        super(Flowerpatch_Embeddings_from_df, self).__init__()
        self.emb_df = emb_df.sort_values(['ID','Track']).reset_index(drop=True) #pd df of precomputed embeddings
        self.imgs_per_track = imgs_per_track # number of images in each track to feed 
        self.get_pair_idx()
        self.data_arr = self.get_numpy_embeddings()

    def __len__(self):
        return len(self.emb_df)
    
    def get_pair_idx(self):
    ### test with making mean pooled reference set, how that compares 
        self.emb_df['Pairs_indices'] = self.emb_df.apply(lambda row: get_track_id_image(self.emb_df,row.name,self.imgs_per_track), axis=1)
        #turn sampled indices into arrays
        self.pair_samples_idxs = self.emb_df["Pairs_indices"].values.flatten()
        
    def get_numpy_embeddings(self):
        '''parse df to get numpy array'''
        #TODO clean this up but should fix data file discrepancies
        i = 0 
        for col_name in self.emb_df.columns:
            if "Feature" in col_name:
                break
            else:
                i+=1
        embs_np = self.emb_df.iloc[:,i:-1].values #exclude track column,and pair indices col at end
        embs_np_n = embs_np / np.linalg.norm(embs_np, axis=1, keepdims=True) #L@ normalization
        self.id_arr = self.emb_df["ID"].values
        return embs_np_n

    def __getitem__(self, idx): 
        embs = self.data_arr[idx]
        pair_idxs = self.pair_samples_idxs[idx]
        selected_idx = np.random.choice(pair_idxs,self.imgs_per_track-1)
        emb_cat = [embs] #TODO: fix this not list 
        for i in range(self.imgs_per_track-1):
            emb_cat.append(self.data_arr[selected_idx[i]])
        #embs = np.ndarray(emb_cat)
        embs =  torch.tensor(emb_cat)
        id = torch.tensor(self.id_arr[idx])
        return {'track_embeddings':embs,'id':id}


############################# Dataset Class for pairs of images #######################################
#
# A dataset that returns a pair of images, and a label of whether they are the same id or not
#
class Flowerpatch_Pairs(Dataset): 
    def __init__(self, df, fname_col, label_col, image_size, split, aug_p = 0.3, pos_pair_prob = 0.5):
        super(Flowerpatch_Pairs, self).__init__()
        self.df = df
        self.df_len = len(df)
        self.fname_col = fname_col # column containing file name or path
        self.label_col = label_col # column containing label/ID
        self.image_size = image_size # image size, for Resize transform
        self.split = split # specifies dataset split (i.e., train vs valid vs test vs ref vs query)
        self.aug_p = aug_p # prob to apply data augmentation methods
        self.pos_pair_prob = pos_pair_prob #probability of getting a positive match
        self.transform = transforms.Compose([transforms.Resize(image_size),
                                             transforms.ToTensor(),
                                            ])
        augmentation_methods = transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(degrees=(0, 2*np.pi)), 
                                                                  transforms.ColorJitter(brightness=0.5, contrast=0.5)]), p=aug_p)
        self.train_transform = transforms.Compose([augmentation_methods,
                                                    transforms.Resize(image_size),
                                                    transforms.ToTensor()]) # include here augmentation techniques
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df.iloc[idx][self.label_col]
        probabilities = [1.0-self.pos_pair_prob, self.pos_pair_prob]
        pair_label = np.random.choice([0,1],p=probabilities)
        #print(pair_label)
        if pair_label == 0:
            negitems = self.df[self.df[self.label_col] != label].sample(n=1)
            pair_img_path = negitems[self.fname_col].values[0]
            #print(pair_img_path)
        else:
            positems = self.df[self.df[self.label_col] == label].sample(n=1)
            pair_img_path = positems[self.fname_col].values[0]
            #print(pair_img_path)
        
        img_path = self.df.iloc[idx][self.fname_col]
        image = Image2.open(img_path)
        pair_image = Image2.open(pair_img_path)
        
        # add transforms with data augmentation if train set
        if self.split == 'train':
            image = self.train_transform(image)
            pair_image = self.train_transform(pair_image)
        else:
            image = self.transform(image)
            pair_image = self.transform(pair_image)

        #convert labels to tensors: 
        label = torch.tensor(label, dtype=torch.float)
        pair_label = torch.tensor(pair_label, dtype=torch.float)
        
        return {'image':image, 'pair': pair_image, 'label':pair_label}


############################# Dataset Class for Pairs of Tracks #######################################
#
# A dataset that returns a pair of tracks, each imgs_per_track long, and a label of whether they are the same id or not
#
class Flowerpatch_Pair_Tracks(Dataset): 
    def __init__(self, df, fname_col, label_col, image_size, split, imgs_per_track=5, aug_p = 0.3, pos_pair_prob = 0.5,track_col = "track"):
        super(Flowerpatch_Pair_Tracks, self).__init__()
        self.df = df
        self.fname_col = fname_col # column containing file name or path
        self.label_col = label_col # column containing label/ID
        self.track_col = track_col
        self.image_size = image_size # image size, for Resize transform #----------------------------#REDUNTANT?
        self.imgs_per_track = imgs_per_track # number of images in each track to feed 
        self.split = split # specifies dataset split (i.e., train vs valid vs test vs ref vs query)
        self.aug_p = aug_p # prob to apply data augmentation methods
        self.pos_pair_prob = pos_pair_prob #probability of getting a positive match
        self.transform = transforms.Compose([transforms.Resize(image_size),
                                             transforms.ToTensor(),
                                            ])
        augmentation_methods = transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(degrees=(0, 2*np.pi)), 
                                                                  transforms.ColorJitter(brightness=0.5, contrast=0.5)]), p=aug_p)
        self.train_transform = transforms.Compose([augmentation_methods,
                                                    transforms.Resize(image_size),
                                                    transforms.ToTensor()]) # include here augmentation techniques
        print("Full:",len(self.df))
        #  filter df to only contain long enough tracks
        self.df = self.df.groupby(self.track_col).filter(lambda x: len(x) > self.imgs_per_track)
        print("With enough imgs per track:",len(self.df))
        #  remove id with only 1 track
        self.df = self.df.groupby(self.label_col).filter(lambda x: len(np.unique(x[self.track_col])) > 1)
        print("With only ids with multiple filtered tracks:",len(self.df))
        self.df_len = len(df)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):            
        img_track =  self.df.iloc[idx][self.track_col] #get track of current sampled
        img_id = self.df.iloc[idx][self.label_col] #get id of current sample
        img_same_tracks = self.df[(self.df[self.label_col] == img_id) & (self.df[self.track_col] == img_track)]

        #print("Length of img_same_tracks before sampling:", len(img_same_tracks))
        if len(img_same_tracks) >= self.imgs_per_track - 1:
            img_same_tracks = img_same_tracks.sample(n=(self.imgs_per_track - 1))
            #print("Successfully sampled", len(img_same_tracks), "items from img_same_tracks.")
        else:
            print("Warning: Not enough items in img_same_tracks to sample.")
            img_same_tracks = img_same_tracks.sample(n=(self.imgs_per_track - 1), replace=True)

        #img_same_tracks = self.df[(self.df[self.label_col] == img_id) & (self.df[self.track_col] == img_track)].sample(n=(self.imgs_per_track-1)) #sample n_imgs -1 from current track 
        img_track_paths = np.concatenate(([self.df.iloc[idx][self.fname_col]],img_same_tracks[self.fname_col])) #append tracks together 
        img_track_imgs = [Image2.open(path) for path in img_track_paths] #open all paths

        probabilities = [1.0-self.pos_pair_prob, self.pos_pair_prob]
        pair_label = np.random.choice([0,1],p=probabilities)

        #print(pair_label)
        if pair_label == 0:
            neg_anchor = self.df[self.df[self.label_col] != img_id].sample(n=1)
            negitem_track = neg_anchor[self.track_col].values[0]
            negitem_id = neg_anchor[self.label_col].values[0]
            negitems = self.df[(self.df[self.label_col] == negitem_id) & (self.df[self.track_col] == negitem_track)]
            if len(negitems) >= self.imgs_per_track:
                negitems = negitems.sample(self.imgs_per_track)
            else:
                print("Warning, not enough samples found for negative track")
                print('Only found',len(negitems),'Unique images in track')
                negitems = negitems.sample(self.imgs_per_track,replace=True)
            #negitems = self.df[(self.df[self.label_col] == negitem_id) & (self.df[self.track_col] == negitem_track)].sample(n=self.imgs_per_track)
            pair_track_paths = negitems[self.fname_col]
            #print(pair_img_path)
        else:
            pos_anchor = self.df[(self.df[self.label_col] == img_id) & (self.df[self.track_col] != img_track)].sample(n=1)
            positem_track = pos_anchor[self.track_col].values[0]
            positem_id = pos_anchor[self.label_col].values[0]
            positems = self.df[(self.df[self.label_col] == positem_id) & (self.df[self.track_col] == positem_track)]
            if len(positems) >= self.imgs_per_track:
                positems = positems.sample(self.imgs_per_track)
            else:
                print("Warning, not enough samples found for positive track")
                print('Only found',len(positems),'Unique images in track')
                positems = positems.sample(self.imgs_per_track,replace=True)
            #positems = self.df[(self.df[self.label_col] == positem_id) & (self.df[self.track_col] == positem_track)].sample(n=self.imgs_per_track)
            pair_track_paths = positems[self.fname_col]

        pair_track_imgs = [Image2.open(pair_path) for pair_path in pair_track_paths]

        # add transforms with data augmentation if train set
        if self.split == 'train':
            images = [self.train_transform(img) for img in img_track_imgs]
            pair_images = [self.train_transform(pair_img) for pair_img in pair_track_imgs]
        else:
            images = [self.transform(img) for img in img_track_imgs]
            pair_images = [self.transform(pair_img) for pair_img in pair_track_imgs]

        #convert label to tensors: 
        pair_label = torch.tensor(pair_label, dtype=torch.float)

        #stack lists of image tensors into one total tensor array
        images = torch.stack(images)
        pair_images = torch.stack(pair_images)
        print('Original sample shape:',images.size(),'Pair shape:',pair_images.size())
        
    
        return {'image_track':images, 'pair_track': pair_images, 'label':pair_label}

#########################################################################################
# A dataset class that returns a pair of tracks but preembedded using specified embedder model
    

############################# Dataset Class for Pairs of Tracks #######################################
#
# A dataset that returns a pair of tracks, each imgs_per_track long, and a label of whether they are the same id or not
#

class Flowerpatch_Pair_Track_Embeddings(Dataset): 
    def __init__(self, df, fname_col, label_col, image_size, split, emb_path, imgs_per_track=5, aug_p = 0.3, pos_pair_prob = 0.5,track_col = "track"):
        super(Flowerpatch_Pair_Track_Embeddings, self).__init__()
        self.df = df
        self.fname_col = fname_col # column containing file name or path
        self.label_col = label_col # column containing label/ID
        self.track_col = track_col
        self.image_size = image_size # image size, for Resize transform #----------------------------#REDUNTANT?
        self.imgs_per_track = imgs_per_track # number of images in each track to feed 
        self.split = split # specifies dataset split (i.e., train vs valid vs test vs ref vs query)
        self.aug_p = aug_p # prob to apply data augmentation methods
        self.pos_pair_prob = pos_pair_prob #probability of getting a positive match
        self.transform = transforms.Compose([transforms.Resize(image_size),
                                             transforms.ToTensor(),
                                            ])
        augmentation_methods = transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(degrees=(0, 2*np.pi)), 
                                                                  transforms.ColorJitter(brightness=0.5, contrast=0.5)]), p=aug_p)
        self.train_transform = transforms.Compose([augmentation_methods,
                                                    transforms.Resize(image_size),
                                                    transforms.ToTensor()]) # include here augmentation techniques
        print("Full:",len(self.df))
        #  filter df to only contain long enough tracks
        self.df = self.df.groupby(self.track_col).filter(lambda x: len(x) > self.imgs_per_track)
        print("With enough imgs per track:",len(self.df))
        #  remove id with only 1 track
        self.df = self.df.groupby(self.label_col).filter(lambda x: len(np.unique(x[self.track_col])) > 1)
        print("With only ids with multiple filtered tracks:",len(self.df))
        self.df_len = len(df)

        #Load embeddor and get embeddings
        self.model_name = os.path.basename(emb_path)
        self.embedder = torch.load(emb_path)
        self.embedder.eval() 
        
        # BUILD DATASET AND DATALOADER
        dataset = Flowerpatch_w_Track(self.df,self.fname_col, self.label_col,self.track_col,self.image_size,self.split)
        bs=32
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False)
        train_embeddings, train_labels, train_tracks = get_embeddings(self.embedder,dataloader)
        print('Train embeddings made with',self.model_name,"shape:",train_embeddings.shape)

                

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):            
        img_track =  self.df.iloc[idx][self.track_col] #get track of current sampled
        img_id = self.df.iloc[idx][self.label_col] #get id of current sample
        img_same_tracks = self.df[(self.df[self.label_col] == img_id) & (self.df[self.track_col] == img_track)]

        #print("Length of img_same_tracks before sampling:", len(img_same_tracks))
        if len(img_same_tracks) >= self.imgs_per_track - 1:
            img_same_tracks = img_same_tracks.sample(n=(self.imgs_per_track - 1))
            #print("Successfully sampled", len(img_same_tracks), "items from img_same_tracks.")
        else:
            print("Warning: Not enough items in img_same_tracks to sample.")
            img_same_tracks = img_same_tracks.sample(n=(self.imgs_per_track - 1), replace=True)

        #img_same_tracks = self.df[(self.df[self.label_col] == img_id) & (self.df[self.track_col] == img_track)].sample(n=(self.imgs_per_track-1)) #sample n_imgs -1 from current track 
        img_track_paths = np.concatenate(([self.df.iloc[idx][self.fname_col]],img_same_tracks[self.fname_col])) #append tracks together 
        img_track_imgs = [Image2.open(path) for path in img_track_paths] #open all paths

        probabilities = [1.0-self.pos_pair_prob, self.pos_pair_prob]
        pair_label = np.random.choice([0,1],p=probabilities)

        #print(pair_label)
        if pair_label == 0:
            neg_anchor = self.df[self.df[self.label_col] != img_id].sample(n=1)
            negitem_track = neg_anchor[self.track_col].values[0]
            negitem_id = neg_anchor[self.label_col].values[0]
            negitems = self.df[(self.df[self.label_col] == negitem_id) & (self.df[self.track_col] == negitem_track)]
            if len(negitems) >= self.imgs_per_track:
                negitems = negitems.sample(self.imgs_per_track)
            else:
                print("Warning, not enough samples found for negative track")
                print('Only found',len(negitems),'Unique images in track')
                negitems = negitems.sample(self.imgs_per_track,replace=True)
            #negitems = self.df[(self.df[self.label_col] == negitem_id) & (self.df[self.track_col] == negitem_track)].sample(n=self.imgs_per_track)
            pair_track_paths = negitems[self.fname_col]
            #print(pair_img_path)
        else:
            pos_anchor = self.df[(self.df[self.label_col] == img_id) & (self.df[self.track_col] != img_track)].sample(n=1)
            positem_track = pos_anchor[self.track_col].values[0]
            positem_id = pos_anchor[self.label_col].values[0]
            positems = self.df[(self.df[self.label_col] == positem_id) & (self.df[self.track_col] == positem_track)]
            if len(positems) >= self.imgs_per_track:
                positems = positems.sample(self.imgs_per_track)
            else:
                print("Warning, not enough samples found for positive track")
                print('Only found',len(positems),'Unique images in track')
                positems = positems.sample(self.imgs_per_track,replace=True)
            #positems = self.df[(self.df[self.label_col] == positem_id) & (self.df[self.track_col] == positem_track)].sample(n=self.imgs_per_track)
            pair_track_paths = positems[self.fname_col]

        pair_track_imgs = [Image2.open(pair_path) for pair_path in pair_track_paths]

        # add transforms with data augmentation if train set
        if self.split == 'train':
            images = [self.train_transform(img) for img in img_track_imgs]
            pair_images = [self.train_transform(pair_img) for pair_img in pair_track_imgs]
        else:
            images = [self.transform(img) for img in img_track_imgs]
            pair_images = [self.transform(pair_img) for pair_img in pair_track_imgs]

        #convert label to tensors: 
        pair_label = torch.tensor(pair_label, dtype=torch.float)

        #stack lists of image tensors into one total tensor array
        images = torch.stack(images)
        pair_images = torch.stack(pair_images)
        print('Original sample shape:',images.size(),'Pair shape:',pair_images.size())
        
    
        return {'image_track':images, 'pair_track': pair_images, 'label':pair_label}




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
def get_dataset(data_config, split,generate_valid=False):
    df = pd.read_csv(data_config['datafiles'][split])
    # only shuffle if train
    if generate_valid == True:
        if split == 'train':

            #sample percentage of training dataset as validation
            valid_num_rows = round(data_config['percent_valid']*len(df))
            valid_rows = df.sample(n=valid_num_rows)

            train_df = df.drop(valid_rows.index)
            train_num = len(train_df)
            valid_df = valid_rows

            print(f"Using {valid_num_rows} samples for validation set")
            print(f"{train_num} total training samples")

            #build dataset and dataloader for modified dataframes
            train_dataset = Flowerpatch(train_df, data_config['fname_col'],
                                        data_config['label_col'], data_config['input_size'], split, data_config['aug_p'])
            train_dataloader = DataLoader(train_dataset, batch_size=data_config['batch_size'], shuffle=True)

            #also build validation dataset
            valid_dataset = Flowerpatch(valid_df, data_config['fname_col'],
                                        data_config['label_col'], data_config['input_size'], split, data_config['aug_p'])
            valid_dataloader = DataLoader(valid_dataset, batch_size=data_config['batch_size'], shuffle=False)

            return (train_dataloader, valid_dataloader)
        if split == 'test':
             #sample percentage of training dataset as validation
            valid_num_rows = round(data_config['percent_reference']*len(df))
            valid_rows = df.sample(n=valid_num_rows)

            train_df = df.drop(valid_rows.index)
            train_num = len(train_df)
            valid_df = valid_rows

            print(f"Using {valid_num_rows} samples for reference set")
            print(f"{train_num} total test samples")

            #build dataset and dataloader for modified dataframes
            train_dataset = Flowerpatch(train_df, data_config['fname_col'],
                                        data_config['label_col'], data_config['input_size'], split, data_config['aug_p'])
            train_dataloader = DataLoader(train_dataset, batch_size=data_config['batch_size'], shuffle=True)

            #also build validation dataset
            valid_dataset = Flowerpatch(valid_df, data_config['fname_col'],
                                        data_config['label_col'], data_config['input_size'], split, data_config['aug_p'])
            valid_dataloader = DataLoader(valid_dataset, batch_size=data_config['batch_size'], shuffle=False)

            return (train_dataloader, valid_dataloader)
    else:
        dataset = Flowerpatch(df, data_config['fname_col'], data_config['label_col'], data_config['input_size'], split, data_config['aug_p'])
        dataloader = DataLoader(dataset, batch_size=data_config['batch_size'], shuffle=False)
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



###################################################################################################
# FUNCTION FOR PREPARING DATASET FOR TRIPLET LOSS FUNCTION #DEPRECATED 
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
