# Import packages and functions

import numpy as np
import pandas as pd
from PIL import Image as Image2
import matplotlib.pyplot as plt
import time
import argparse
import yaml
import pickle
from datetime import datetime

import torch
from transformers import ViTFeatureExtractor
from pytorch_data import *
from pytorch_models import *
from sklearn.metrics import confusion_matrix

import torch
import sys
#sys.path.insert(1, '/home/lmeyers/beeid_clean_luke/PYTORCH_CODE/')
#sys.path.insert(2, '/home/lmeyers/beeid_clean_luke/KEY_CODE_FILES/')

from pytorch_resnet50_conv3 import resnet50_convstage3
from data import prepare_for_triplet_loss

import numpy as np
import pandas as pd
from PIL import Image as Image2
import matplotlib.pyplot as plt

import torch.nn.parallel
import sys 
import os
import wandb

from torcheval.metrics.functional import binary_accuracy, binary_precision,binary_recall
from torcheval.metrics.functional import binary_f1_score
import torch
from transformers import ViTFeatureExtractor
from pytorch_data import *
from pytorch_models import *
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

 #load config file params:
config_file = '/home/lmeyers/ReID_complete/MTL_test_10_5_config.yml'
verbose = True

try:
    with open(config_file) as f:
        config = yaml.safe_load(f)
    model_config = config['model_settings'] # settings for model building
    train_config = config['train_settings'] # settings for model training
    data_config = config['data_settings'] # settings for data loading
    eval_config = config['eval_settings'] # settings for evaluation
    torch_seed = config['torch_seed']
    verbose = config['verbose']
except Exception as e:
    print('ERROR - unable to open experiment config file. Terminating.')
    print('Exception msg:',e)
if verbose:
    # ADD PRINT OF DATE AND TIME
    now = datetime.now() # current date and time
    dt = now.strftime("%y-%m-%d %H:%M")
    print(f'Date and time when this experiment was started: {dt}')
    print(f'Date and time when this experiment was started: {dt}')
    print("Data Settings:")
    print(data_config)
    print("Train Settings:")
    print(train_config)
    print("Model Settings:")
    print(model_config)

# Set GPU to use
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

## load data

# setting torch seed
torch.manual_seed(torch_seed)

train_fname = data_config['datafiles']['train']
df_train = pd.read_csv(train_fname)
dft_train = prepare_for_triplet_loss(df_train, data_config['label_col'], data_config['fname_col'])



# BUILD DATASET AND DATALOADER
train_dataset = MultiTaskData(dft_train, 'filename', 'label',data_config['input_size'],'train','/home/lmeyers/ReID_complete/summer_2023_v3_color_map.json')
bs=64
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=False)
batch = next(iter(train_dataloader))

if verbose:
    try:
        batch = next(iter(train_dataloader))
        print(f'Batch image shape: {batch["image"].size()}')
        print(f'Batch label shape: {batch["label"].size()}')
        print(f'Batch color_label shape: {batch["color_label"].size()}')
    except Exception as e:
        print('ERROR - could not print out batch properties')
        print(f'Error msg: {e}')

## Build model and load to device:
num_classes = model_config['num_labels']
base_model = build_model(model_config) #returns backbone specified in yml

class IdentityHead(nn.Module):
    def forward(self, x):
        # The forward pass simply returns the input as is
        return x

class ColorDetectHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super(ColorDetectHead, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

 
class MultiHeadModel(nn.Module):
    def __init__(self, base_model):
        super(MultiHeadModel, self).__init__()
        self.base_model = base_model
        self.identity_head = IdentityHead()
        self.color_detection_head = ColorDetectHead(128,model_config['color_dim'])

    def forward(self, x):
        features = self.base_model(x)
        identity_output = self.identity_head(features)
        color_output = self.color_detection_head(features)
        return identity_output, color_output

# Build Multitask model utilizing imported backbone
MTL_model = MultiHeadModel(base_model)


optimizer = torch.optim.Adam(MTL_model.parameters(), lr=train_config['learning_rate'])

# Send to GPU 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
if verbose:
    print(f'Device: {device}')
MTL_model.to(device)

color_detect_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(4))
reid_loss_fn = losses.TripletMarginLoss(train_config['margin'], distance = CosineSimilarity())

#DATA EVALUATION FUNCTIONS, may be unnecessary 

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


#########################################################################################
# FUNCTION TO PERFORM KNN EVALUATION
#
def knn_evaluation(train_images, train_labels, test_images, test_labels, n_neighbors, per_class=True, conf_matrix=True):
    # BUILD KNN MODEL AND PREDICT
    results = {}
    print(f'Training kNN classifier with k={n_neighbors}')
    my_knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')
    my_knn.fit(train_images, train_labels)
    knn_pred = my_knn.predict(test_images)
    knn_acc = np.round(np.sum([1 for pred, label in zip(knn_pred, test_labels) if pred == label])/test_labels.shape[0],4)
    print(f'{n_neighbors}NN test accuracy: {knn_acc}')
    # store results
    results['n_neighbors'] = n_neighbors
    results['knn'] = knn_acc
    label_list = np.unique(train_labels)
    results['label_list'] = label_list
    if per_class:
        knn_class = np.zeros(len(label_list))
        print(f'\nPer label {n_neighbors}NN test accuracy:')
        for k, label in enumerate(label_list):
            mask = test_labels == label
            knn_class[k] = np.round(np.sum(knn_pred[mask]==test_labels[mask])/np.sum(mask),4)
            print(f'{label}\t{knn_class[k]:.2f}')
        # store results
        results['knn_class'] = knn_class
    if conf_matrix:
        knn_conf = confusion_matrix(test_labels, knn_pred)
        results['knn_conf'] = knn_conf
        print('\nPrinting Confusion Matrix:')
        print(results['knn_conf'])
    return results
#########################################################################################

##########################################################################################
# FUNCTION TO GET EMBEDDINGS AND LABELS FOR EVALUATING MODEL
def get_predictions(model, dataloader, loss_fn, device, feature_extractor=None):
    predictions = []
    all_labels = []
    loss = 0.0
    with torch.no_grad():
        for k, batch in enumerate(dataloader):
            if feature_extractor is None:
                images = batch['image']
            else:
                images = [transforms.functional.to_pil_image(x) for x in batch['image']]
                images = np.concatenate([feature_extractor(x)['pixel_values'] for x in images])
                images = torch.tensor(images, dtype=torch.float)
            labels = batch['label'].to(device)
            outputs = model(images.to(device))
            loss += loss_fn(outputs, labels).detach().cpu().numpy()
            predictions += list(outputs.detach().cpu().numpy())
            all_labels += list(labels.detach().cpu().numpy())
    predictions = np.array(predictions)
    all_labels = np.array(all_labels)
    loss/=k
    return predictions, all_labels, loss
##########################################################################################

output = torch.tensor([[0.1836, 0.1990, 0.1784, 0.1853, 0.4892, 0.4804, 0.4951, 0.4842, 0.5116,
        0.5112, 0.7208, 0.5219, 0.5184],[0.1836, 0.1990, 0.1784, 0.1853, 0.4892, 0.4804, 0.4951, 0.4842, 0.5116,
        0.5112, 0.7208, 0.5219, 0.5184]])

label = torch.tensor([[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]])


import torch

def calculate_confusion_matrix(predictions, ground_truth):
    """
    Calculate confusion matrix for binary classification.

    Args:
        predictions (torch.Tensor): Tensor of binary predictions (0 or 1).
        ground_truth (torch.Tensor): Tensor of ground truth labels (0 or 1).

    Returns:
        dict: A dictionary with TP, TN, FP, and FN counts.
    """
    # Ensure both inputs are tensors
    if not isinstance(predictions, torch.Tensor) or not isinstance(ground_truth, torch.Tensor):
        raise ValueError("Both inputs must be PyTorch tensors.")

    # Ensure the tensors have the same shape
    if predictions.shape != ground_truth.shape:
        raise ValueError("Input tensors must have the same shape.")

    # Round prediction to be binary like truth
    predictions = torch.round(predictions)

    # Calculate confusion matrix counts
    tp = torch.sum((predictions == 1.0) & (ground_truth == 1.0)).item()
    tn = torch.sum((predictions == 0.0) & (ground_truth == 0.0)).item()
    fp = torch.sum((predictions == 1.0) & (ground_truth == 0.0)).item()
    fn = torch.sum((predictions == 0.0) & (ground_truth == 1.0)).item()

    # Create and return the dictionary
    confusion_matrix = {
        'True Positives (TP)': tp,
        'True Negatives (TN)': tn,
        'False Positives (FP)': fp,
        'False Negatives (FN)': fn
    }

    return confusion_matrix

############################################################################################
# FUNCTION TO EVAL TWO COLOR MAPS, INCLUDING PRECISION, RECALL, F1 AND PER CLASS ACC

def getMetrics(predictions,truth):
    '''
    A function to return 3 evaluation metrics on each sample.
    
    Args:
        predictions (torch.Tensor): Tensor of binary predictions (0 or 1).
        truth (torch.Tensor): Tensor of ground truth labels (0 or 1).
    Returns:
        accuracy(torch.Tensor): Tensor of average # of classes that are predicted correctly (TP+TN/All)
        precision(torch.Tensor): tensor of average precision
        recall(torch.Tensor): Tensor of average recall
    '''
    if not isinstance(predictions, torch.Tensor) or not isinstance(truth, torch.Tensor):
        raise ValueError("Both inputs must be PyTorch tensors.")

    # Ensure the tensors have the same shape
    if predictions.shape != truth.shape:
        raise ValueError("Input tensors must have the same shape.")

    acc = []
    precision = []
    recall = []
    for t in range(len(predictions)):
        acc.append(binary_accuracy(torch.round(predictions[t]),truth[t]))
        precision.append(binary_precision(torch.round(predictions[t]),truth[t]))
        recall.append(binary_recall(torch.round(predictions[t]).long(),truth[t].long()))
    acc = torch.mean(torch.tensor(acc))
    precision = torch.tensor(precision).mean()
    recall = torch.tensor(recall).mean()
    return acc, precision, recall


### ----- actually calling the training

resume_training = False

#"""
if resume_training == True: 
    experiment = wandb.init(project="MTL_summer_data", entity="lqmeyers",resume=True,id='w42wxecm')
else:
    experiment = wandb.init(project="MTL_summer_data", entity="lqmeyers")
#"""

# load latest saved checkpoint if resuming a failed run
#"""
if resume_training == True: 
    saved = os.listdir(os.path.dirname(model_config['model_path'])+r'/checkpoints/')
    check_array = []
    for f in saved:
        check_array.append(f[:-4])
    check_array = np.array(check_array,dtype=np.int64)
    #most_recent_epoch = np.max(check_array) #find most recent checkpoint
    most_recent_epoch = 50 #setting manually from known interuption
    print(f'Resuming training from saved epoch: {most_recent_epoch}')
    most_recent_model = os.path.dirname(model_config['model_path'])+r'/checkpoints/'+str(most_recent_epoch)+'.pth'
    print(f'Loading saved checkpoint model {most_recent_model}')
    MTL_model = torch.load(most_recent_model)
#"""



# if resuming training set epoch number
if resume_training == True:
    epoch_range = range(train_config['num_epochs'])[most_recent_epoch:]
else:
    epoch_range = range(train_config['num_epochs'])

    # Train the model
if verbose:
    print('Training model...')

print_k = train_config['print_k']
start = time.time()
for epoch in epoch_range: 
    running_loss = 0.0
    for k, data in enumerate(train_dataloader):
        images = data['image'].to(device)
        labels = data['label'].to(device)
        #print(labels.size())
        color_labels = data['color_label'].to(device)
        #print(color_labels.size())
        optimizer.zero_grad()
        features, color_pred = MTL_model(images)
        reid_loss = reid_loss_fn(features,labels)
        color_detect_loss = color_detect_loss_fn(color_pred,color_labels)
        total_loss = model_config['loss_weights'][0]*reid_loss+model_config['loss_weights'][1]*color_detect_loss
        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item()
        #"""
        experiment.log({
            'train loss': total_loss.item(),
            'reid loss': reid_loss,
            'color detect loss ': color_detect_loss,
            'epoch': epoch+1
            #'learning rate' : lr
        })
        #"""
        if (k+1)%print_k == 0:
            with torch.no_grad():
                train_acc, train_prec, train_recall = getMetrics(color_pred,color_labels)
                #valid_predictions, valid_labels, valid_loss = get_predictions(MTL_model, valid_dataloader, color_detect_loss_fn, device)
                #valid_acc, valid_prec, valid_recall = getMetrics(torch.tensor(valid_predictions),torch.tensor(valid_labels))
                #"""
                experiment.log({'epoch':(epoch+1),
                                'train loss': (running_loss/print_k),
                                'train accuracy':train_acc,
                                'train precision':train_prec,
                                'train recall': train_recall,
                                #'validation loss': valid_loss,
                                #'validation accuracy': valid_acc,
                                #'validation precision': valid_prec,
                                #'valdiation recall': valid_recall
                                })
                                #"""
                print(f'[{epoch + 1}, {k + 1:5d}] loss: {running_loss/print_k:.4f} | train_acc: {train_acc:.4f} | train_recall: {train_recall:.4f}') #| val_loss: {valid_loss:.4f} | val_acc: {valid_acc:.4f} | val_recall{valid_recall:.4f}')
                running_loss= 0.0
        #"""

        """
        if (k+1)%print_k == 0:
            with torch.no_grad():
                valid_outputs, valid_labels, valid_loss = get_embeddings(model, valid_dataloader, loss_fn, miner, device, feature_extractor)
                print(f'[{epoch + 1}, {k + 1:5d}] train_loss: {running_loss/print_k:.4f} | val_loss: {valid_loss:.4f}')
                running_loss=0.0
                #scheduler.step(valid_loss)
                #current_lr = optimizer.param_groups[0]['lr']
                experiment.log({'valid loss': valid_loss, })
                                # 'learning rate': current_lr})
        """
    if epoch % int(train_config['save_checkpoint_freq']) == 0 or epoch+1 == train_config['num_epochs']:
            if os.path.dirname(model_config['model_path']) is not None:
                print('Saving checkpoint',epoch)
                if not os.path.exists(os.path.dirname(model_config['model_path'])+r'/checkpoints/'):
                    os.mkdir(os.path.dirname(model_config['model_path'])+r'/checkpoints/')
                torch.save(MTL_model,(os.path.dirname(model_config['model_path'])+r'/checkpoints/'+str(epoch)+".pth"))

stop = time.time()
experiment.finish()
print(f'Total train time: {(stop-start)/60}min')

if model_config['model_path'] is not None:
            print('Saving model...')
            torch.save(MTL_model, model_config['model_path'])
else:
    print('model_path not provided. Not saving model')

####--------- Perform Evaluation---------------

## load test dataset 
test_fname = data_config['datafiles']['test']
df_test = pd.read_csv(test_fname)
dft_test = prepare_for_triplet_loss(df_test,data_config['label_col'],data_config['fname_col'])

# BUILD DATASET AND DATALOADER
test_dataset = MultiTaskData(dft_test, 'filename', 'label',data_config['input_size'],'test','/home/lmeyers/ReID_complete/summer_2023_v3_color_map.json')
bs=64
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False)
batch = next(iter(test_dataloader))

#load saved model:
#model = torch.load('/home/lmeyers/ReID_complete/wandb/latest-run/files/model.pth')
#MTL_model = torch.load('/home/lmeyers/ReID_complete/checkpoints/100.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
if verbose:
    print(f'Device: {device}')
MTL_model.to(device)

# PERFORM INFERENCE
with torch.no_grad():
    acc = []
    prec = []
    rec = []
    vals = {}
    corr = []
    for k, data in enumerate(test_dataloader):
        if verbose: 
            print('Performing Inference on Batch',k)
        images = data['image'].to(device)
        # get labels
        labels = data['color_label'].to(device)
        features, color_preds = MTL_model(images)
        outputs = color_preds
        #matches = (torch.round(outputs) == labels)
        #print(matches.shape)
        #correct = np.all(matches,axis=1,out=None)
        #print(correct.shape)
        #matches = torch.logical_and(torch.round(outputs),labels)
        predictions = torch.round(outputs)
        mapping_array = np.zeros_like(predictions.cpu())  # Initialize with zero TN
        # Set elements to 1 where both binary arrays have value 1
        mapping_array[(predictions.cpu() == 1) & (labels.cpu() == 1)] = 1 #TP
        mapping_array[(predictions.cpu() == 1) & (labels.cpu() == 0)] = 2 #FP
        mapping_array[(predictions.cpu() == 0) & (labels.cpu() == 1)] = 3 #FN
        #accuracy = (matches.float().mean())
        accuracy, precision, recall = getMetrics(outputs,labels)
        acc.append(accuracy)
        prec.append(precision)
        rec.append(recall)
        corrects = [2 and 3 not in sample for sample in mapping_array]
        corr = np.concatenate((corr,corrects))
        for i in range(len(labels[0])):
            values_at_index = [subarray[i] for subarray in mapping_array]
            #print('Batch',k,'Class',i,values_at_index)
            if k == 0:
                vals[i] = values_at_index  
            else:
                vals[i] = np.concatenate((vals[i],values_at_index))
    for c in vals:
            TP = torch.sum(torch.tensor(vals[c]) == 1)
            TN = torch.sum(torch.tensor(vals[c]) == 0)
            FP = torch.sum(torch.tensor(vals[c]) == 2)
            FN = torch.sum(torch.tensor(vals[c]) == 3)
            All = torch.tensor(len(vals[c]))
            class_acc = (TP + TN)/ All
            class_prec=  (TP)/(TP+FP)
            class_rec = (TP)/(TP+FN)
            print('Class',c,'TP:',int(TP),'FN:',int(FN),'FP:',int(FP),'TN:',int(TN))#'Acc:',torch.tensor(class_acc).float().mean(),'Precision:',torch.tensor(class_prec).float().mean(),'Recall:',torch.tensor(class_rec).float().mean(),)
    print('Total Correct Samples',np.sum(corr),"out of",len(corr),'=',(np.sum(corr)/len(corr)))
    print("Total Acc:",torch.tensor(acc).mean())
    print("Total Precision:",torch.tensor(prec).mean())
    print("Total Recall:",torch.tensor(rec).mean())
       