import numpy as np
import pandas as pd
from PIL import Image as Image2
import matplotlib.pyplot as plt
import time
import argparse
import yaml
import pickle
from datetime import datetime
import gc
import os
import sys 
import wandb
import pickle
sys.path.insert(0,"../")

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


print("finished imports")

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
    print(f"Training kNN classifier with k=1")
    my_knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
    my_knn.fit(train_images, train_labels)
    knn_pred = my_knn.predict(test_images)
    knn_acc = np.round(np.sum([1 for pred, label in zip(knn_pred, test_labels) if pred == label])/test_labels.shape[0],4)
    print(f"1NN test accuracy: {knn_acc}")
    # store results
    results['1NN_acc'] = knn_acc

    print(f"Training kNN classifier with k=3")
    my_knn = KNeighborsClassifier(n_neighbors=3, metric='cosine')
    my_knn.fit(train_images, train_labels)
    knn_pred = my_knn.predict(test_images)
    knn_acc = np.round(np.sum([1 for pred, label in zip(knn_pred, test_labels) if pred == label])/test_labels.shape[0],4)
    print(f'3NN test accuracy: {knn_acc}')
    # store results
    results['3NN_acc'] = knn_acc

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


def eval(config_file):
    
    try:
        with open(config_file) as f:
            config = yaml.safe_load(f)
        train_config = config['train_settings']
        model_config = config['model_settings'] # settings for model building
        data_config = config['data_settings'] # settings for data loading
        eval_config = config['eval_settings'] # settings for evaluation
        torch_seed = config['torch_seed']
        verbose = config['verbose']
    except Exception as e:
        print('ERROR - unable to open experiment config file. Terminating.')
        print('Exception msg:',e)
        return -1
    

    
    if verbose:
        now = datetime.now() # current date and time
        dt = now.strftime("%y-%m-%d %H:%M")
        print(f'Date and time when this experiment was started: {dt}')
        print("Data Settings:")
        print(data_config)
        print("Model Settings:")
        print(model_config)

    #SET GPU TO USE
    os.environ["CUDA_VISIBLE_DEVICES"]=str(train_config['gpu'])
    if verbose:
        print('Using GPU',train_config['gpu'])

    # setting torch seed
    torch.manual_seed(torch_seed)
    
  
    # build model
    if verbose:
        print('Building model....',model_config['model_path'])
    model = torch.load(model_config['model_path'])
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'])
    # Initialize optimizer and scheduler
    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.75, verbose=True,min_lr = 1e-5)

    #miner = miners.MultiSimilarityMiner()
    miner = miners.TripletMarginMiner(margin=train_config['margin'], type_of_triplets="semihard", distance = CosineSimilarity())
    miner_type = "semihard"
    loss_fn = losses.TripletMarginLoss(train_config['margin'], distance = CosineSimilarity())
    if verbose:
        print('Loss:',loss_fn)

    # load VIT feature extractor if needed
    if model_config['model_class'].startswith('vit'):
        if verbose:
            print('Getting ViT feature extractor...')
        model_name = 'google/vit-base-patch16-224-in21k'
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    else:
        feature_extractor = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f'Found device: {device}')
    model.to(device)

    

    num_epochs_no_improvement = 0
    

    start = time.time()
   

    # evaluate on test set using KNN
    if verbose:
        print('Evaluating model...')
    model.eval()
    # load "reference" for training KNN
    #       - Closed Setting: same as training set
    #       - Open Setting: the actual reference set
    # load "query" for testing
    #       - Closed Setting: test set
    #       - Open SettingL the actual query set
    if model_config['model_class'].startswith('swin3d'):
        reference_dataloader = get_track_dataset(data_config, 'reference')
        test_dataloader = get_track_dataset(data_config, 'query')
    else:
        if data_config['sample_reference'] == True:
            test_dataloader, reference_dataloader = get_dataset(data_config, 'test',generate_valid=True) #generate valid automatically
        else:
            reference_dataloader = get_dataset(data_config, 'reference')
            test_dataloader = get_dataset(data_config, 'query')
    if verbose:
        print('generating embeddings')
    reference_embeddings, reference_labels, reference_loss = get_embeddings(model, reference_dataloader, loss_fn, miner, device, feature_extractor)
    test_embeddings, test_labels, test_loss = get_embeddings(model, test_dataloader, loss_fn, miner, device, feature_extractor)

    # Convert query/test labels to match referene labels if necessary 
    b1_to_b2 = {10: 74, 11: 75, 15: 79, 14: 78, 12: 76, 16: 80, 13: 77, 9: 73, 18: 82, 19: 83, 23: 87, 22: 86, 20: 84, 24: 88, 21: 85, 17: 81, 50: 114, 51: 115, 55: 119, 54: 118, 52: 116, 56: 120, 53: 117, 49: 113, 42: 106, 43: 107, 47: 111, 46: 110, 44: 108, 48: 112, 45: 109, 41: 105, 26: 90, 27: 91, 31: 95, 30: 94, 28: 92, 32: 96, 29: 93, 25: 89, 58: 122, 59: 123, 63: 127, 62: 126, 60: 124, 64: 128, 61: 125, 57: 121, 34: 98, 35: 99, 39: 103, 38: 102, 36: 100, 40: 104, 37: 101, 33: 97, 2: 66, 3: 67, 7: 71, 6: 70, 68: 68, 8: 72, 5: 69, 1: 65}
    b2_to_b1 = {74: 10, 75: 11, 79: 15, 78: 14, 76: 12, 80: 16, 77: 13, 73: 9, 82: 18, 83: 19, 87: 23, 86: 22, 84: 20, 88: 24, 85: 21, 81: 17, 114: 50, 115: 51, 119: 55, 118: 54, 116: 52, 120: 56, 117: 53, 113: 49, 106: 42, 107: 43, 111: 47, 110: 46, 108: 44, 112: 48, 109: 45, 105: 41, 90: 26, 91: 27, 95: 31, 94: 30, 92: 28, 96: 32, 93: 29, 89: 25, 122: 58, 123: 59, 127: 63, 126: 62, 124: 60, 128: 64, 125: 61, 121: 57, 98: 34, 99: 35, 103: 39, 102: 38, 100: 36, 104: 40, 101: 37, 97: 33, 66: 2, 67: 3, 71: 7, 70: 6, 68: 68, 72: 8, 69: 5, 65: 1}

#     reference_data_batch = os.path.dirname(data_config['datafiles']['reference'])[-1:]
#     query_data_batch = os.path.dirname(data_config['datafiles']['query'])[-1:]

#     if reference_data_batch != query_data_batch and data_config['label_col'] != 'color_num':
#         if reference_data_batch > query_data_batch:
#             for i in range(len(test_labels)):
#                 test_labels[i] = b1_to_b2[test_labels[i]]
#         else: 
#             for i in range(len(test_labels)):
#                 test_labels[i] = b2_to_b1[test_labels[i]]
    
    print(f'Reference (or Train) Loss: {reference_loss:.4f}')
    print('Reference size:',reference_embeddings.shape)
    print(f'Test (or Query) Loss: {test_loss:.4f}')
    print('Test (or Query) size:',test_embeddings.shape)

    results = knn_evaluation(reference_embeddings, reference_labels, test_embeddings, test_labels, 
                            eval_config['n_neighbors'], eval_config['per_class'], eval_config['conf_matrix'])
    stop = time.time()
    duration = (stop-start)/60
    print(f'Total eval time: {duration}min')
    
    # Adding other metrics to results to pass to csv
    results['test_loss'] = test_loss
    # results['wandb_id'] = experiment.id

    # results['start_time'] = experiment.start_time
    # results['train_time'] = duration
    # results['stop_epoch'] = stop_epoch



    # Save results to temporary file
    with open(eval_config['pickle_file'],'wb') as fi:
        pickle.dump(results,fi)



print("beginning execution")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="yaml file with experiment settings", type=str)
    args = parser.parse_args()
    eval(args.config_file)
