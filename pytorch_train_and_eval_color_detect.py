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
import wandb
import pickle

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

from torcheval.metrics.functional import binary_accuracy, binary_precision,binary_recall

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


def train_and_eval(config_file):
    
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
        return -1
    
    resume_training = train_config['wandb_resume']

    #initialize wandb logging
    if resume_training == True: 
        experiment = wandb.init(project= train_config["wandb_project_name"],entity=train_config['wandb_entity_name'],resume=True,id=train_config['wandb_run_id'],dir=train_config['wandb_dir_path'])
    else:
        experiment = wandb.init(project= train_config["wandb_project_name"],entity=train_config['wandb_entity_name'],dir=train_config['wandb_dir_path'])
    

    if verbose:
        # ADD PRINT OF DATE AND TIME
        now = datetime.now() # current date and time
        dt = now.strftime("%y-%m-%d %H:%M")
        print(f'Date and time when this experiment was started: {dt}')
        print("Data Settings:")
        print(data_config)
        print("Train Settings:")
        print(train_config)
        print("Model Settings:")
        print(model_config)
    
    #SET GPU TO USE
    os.environ["CUDA_VISIBLE_DEVICES"]=str(train_config['gpu'])
    if verbose:
        print('Using GPU',train_config['gpu'])

    # setting torch seed
    torch.manual_seed(torch_seed)

    #CREATE TRAIN DATALOADER 
    if verbose:
            print('Creating train and valid dataloaders...')
    # READ DATAFRAMES 
    train_fname = data_config['datafiles']['train']
    df_train = pd.read_csv(train_fname)
    dft_train = prepare_for_triplet_loss(df_train, data_config['label_col'], data_config['fname_col'])

    # BUILD DATASET AND DATALOADER
    train_dataset = ColorMap_w_Order(dft_train, 'filename', 'label',data_config['input_size'],'train',data_config['datafiles']['color_map'])
    bs=64
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
    batch = next(iter(train_dataloader))
    
    if verbose:
        try:
            batch = next(iter(train_dataloader))
            print(f'Batch image shape: {batch["image"].size()}')
            print(f'Batch label shape: {batch["label"].size()}')
        except Exception as e:
            print('ERROR - could not print out batch properties')
            print(f'Error msg: {e}')
    
    #CREATE VALID DATALOADER
    ## load validation dataset 
    valid_fname = data_config['datafiles']['valid']
    df_valid = pd.read_csv(valid_fname)
    dft_valid = prepare_for_triplet_loss(df_valid, data_config['label_col'], data_config['fname_col'])

    # BUILD DATASET AND DATALOADER
    valid_dataset = ColorMap_w_Order(dft_valid, 'filename', 'label',data_config['input_size'],'valid',data_config['datafiles']['color_map'])
    bs=64
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, shuffle=False)
    batch = next(iter(valid_dataloader))

    ## Build model and load to device:
    num_classes = model_config['num_labels']
    base_model = build_model(model_config)

    #add color class detector output to model:
    model = nn.Sequential(
        base_model,
        nn.Linear(128, num_classes*2),
        #nn.Sigmoid()
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'])
    
    w = train_config['pos_weight']
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w))
    if verbose:
        print('Loss:',loss_fn,'Positive Weight:',w)

    # LOAD TO DEVICE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    if verbose:
        print(f'Device: {device}')
    model.to(device)

    # load latest saved checkpoint if resuming a failed run
    if resume_training == True: 
        saved = os.listdir(os.path.dirname(model_config['model_path'])+r'/checkpoints/')
        check_array = []
        for f in saved:
            check_array.append(f[:-4])
        check_array = np.array(check_array,dtype=np.int64)
        #most_recent_epoch = np.max(check_array) #find most recent checkpoint
        most_recent_epoch =  train_config['checkpoint_to_load']
        print(f'Resuming training from saved epoch: {most_recent_epoch}')
        most_recent_model = os.path.dirname(model_config['model_path'])+r'/checkpoints/'+str(most_recent_epoch)+'.pth'
        print(f'Loading saved checkpoint model {most_recent_model}')
        model = torch.load(most_recent_model)


    # if resuming training set epoch number
    if resume_training == True:
        epoch_range = range(train_config['num_epochs'])[most_recent_epoch:]
    else:
        epoch_range = range(train_config['num_epochs'])


    # Initialize early stopping variables
    best_valid_loss = 0
    valid_loss = 'Null'
    num_epochs_no_improvement = 0
    check_for_early_stopping = train_config['early_stopping']
    consecutive_epochs = train_config['early_stop_consecutive_epochs']
    stop_early = False
    

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
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                #"""
                experiment.log({
                    'train loss': loss.item(),
                    'epoch': epoch+1
                    #'learning rate' : lr
                })
                #"""
                #if (k+1)%print_k == 0:
                
        with torch.no_grad():
            train_acc, train_prec, train_recall = getMetrics(outputs,labels)
            valid_predictions, valid_labels, valid_loss = get_predictions(model, valid_dataloader, loss_fn, device)
            valid_predictions = np.round(valid_predictions)
            v_mapping_array = np.zeros_like(valid_predictions)  # Initialize with zero TN
            # Set elements to 1 where both binary arrays have value 1
            v_mapping_array[(valid_predictions == 1) & (valid_labels == 1)] = 1 #TP
            v_mapping_array[(valid_predictions == 1) & (valid_labels == 0)] = 2 #FP
            v_mapping_array[(valid_predictions == 0) & (valid_labels == 1)] = 3 #FN
            
            corrects = [2 and 3 not in sample for sample in v_mapping_array]

            valid_corr = np.sum(corrects)/len(valid_predictions)

            valid_acc, valid_prec, valid_recall = getMetrics(torch.tensor(valid_predictions),torch.tensor(valid_labels))
            valid_f1 = 2*((valid_prec*valid_acc)/(valid_prec+valid_acc))
            experiment.log({'epoch':(epoch+1),
                            'train loss': (running_loss/print_k),
                            'train accuracy':train_acc,
                            'train precision':train_prec,
                            'train recall': train_recall,
                            'valid percent correct': valid_corr,
                            'validation loss': valid_loss,
                            'validation accuracy': valid_acc,
                            'validation precision': valid_prec,
                            'valdiation recall': valid_recall,
                            'f1':valid_f1})
            print(f'[{epoch + 1}, {k + 1:5d}] loss: {running_loss/print_k:.4f} | train_acc: {train_acc:.4f} | train_recall: {train_recall:.4f}| val_loss: {valid_loss:.4f} | val_acc: {valid_acc:.4f} | val_recall: {valid_recall:.4f} | f1: {valid_f1}')
            running_loss= 0.0

        """             
        # Check if validation loss has improved
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            num_epochs_no_improvement = 0
        else:
            num_epochs_no_improvement += 1
        """
        if valid_corr > best_valid_loss:
            best_valid_loss = valid_corr
            best_model = model
            num_epochs_no_improvement = 0
        else:
            num_epochs_no_improvement += 1

        # Check if early stopping condition is met
        if check_for_early_stopping == True:
            if num_epochs_no_improvement >= consecutive_epochs:
                print(f'Early stopping at epoch {epoch+1} due to no improvement in validation loss for {consecutive_epochs} consecutive epochs')
                stop_epoch = epoch+1
                stop_early = True 
        # Breaks Epoch iteration to stop training early
        # will only be true if checking for early stopping is enabled                     
        if stop_early == True:
            break
    
        if epoch % train_config['save_checkpoint_freq'] == 0 or (epoch-1) == train_config['num_epochs']:
            if os.path.dirname(model_config['model_path']) is not None:
                print('Saving checkpoint',epoch)
                if not os.path.exists(os.path.dirname(model_config['model_path'])+r'/checkpoints/'):
                    os.mkdir(os.path.dirname(model_config['model_path'])+r'/checkpoints/')
                torch.save(model,(os.path.dirname(model_config['model_path'])+r'/checkpoints/'+str(epoch)+".pth"))
    model = best_model
    stop_epoch = epoch+1
    stop = time.time()
    duration = (stop-start)/60
    print(f'Total train time: {duration}min')

    # evaluate on test set
    if verbose:
        print('Evaluating model...')
   
    ## load test dataset 
    test_fname = data_config['datafiles']['test']
    df_test = pd.read_csv(test_fname)

    dft_test = prepare_for_triplet_loss(df_test, data_config['label_col'], data_config['fname_col'])

    # BUILD DATASET AND DATALOADER
    test_dataset = ColorMap_w_Order(dft_test, 'filename', 'label',data_config['input_size'],'test',data_config['datafiles']['color_map'])
    bs=64
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False)
    batch = next(iter(test_dataloader))

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
            labels = data['label'].to(device)
            outputs = model(images)
            #matches = (torch.round(outputs) == labels)
            #print(matches.shape)
            #correct = np.all(matches,axis=1,out=None)
            #print(correct.shape)
            #matches = torch.logical_and(torch.round(outputs),labels)
            #predictions = torch.round(outputs) #larger than 0
            predictions = torch.where(outputs > 0,1, 0)
            mapping_array = np.zeros_like(predictions.cpu())  # Initialize with zero TN
            # Set elements to 1 where both binary arrays have value 1
            labels_cpu = labels.cpu()
            #predictions = 
            mapping_array[(predictions.cpu() == 1) & (labels_cpu == 1)] = 1 #TP
            mapping_array[(predictions.cpu() == 1) & (labels_cpu == 0)] = 2 #FP
            mapping_array[(predictions.cpu() == 0) & (labels_cpu == 1)] = 3 #FN
            #accuracy = (matches.float().mean())
            accuracy, precision, recall = getMetrics(outputs,labels)
            acc.append(accuracy)
            prec.append(precision)
            rec.append(recall)
            corrects = [2 and 3 not in sample for sample in mapping_array] #in is expensive (replace!)
            corr = np.concatenate((corr,corrects))
            for i in range(len(labels[0])):
                values_at_index = [subarray[i] for subarray in mapping_array]
                #print('Batch',k,'Class',i,values_at_index)
                if k == 0:
                    vals[i] = values_at_index  
                else:
                    vals[i] = np.concatenate((vals[i],values_at_index))
        class_dict = {}
        for c in vals:
                TP = torch.sum(torch.tensor(vals[c]) == 1)
                TN = torch.sum(torch.tensor(vals[c]) == 0)
                FP = torch.sum(torch.tensor(vals[c]) == 2)
                FN = torch.sum(torch.tensor(vals[c]) == 3)
                All = torch.tensor(len(vals[c]))
                class_acc = (TP + TN)/ All
                class_prec=  (TP)/(TP+FP)
                class_rec = (TP)/(TP+FN)
                class_dict[c]={'acc:':torch.tensor(class_acc).float().mean(),'precision:':torch.tensor(class_prec).float().mean(),'recall:':torch.tensor(class_rec).float().mean()}
                print('Class',c,'TP:',int(TP),'FN:',int(FN),'FP:',int(FP),'TN:',int(TN),'Acc:',torch.tensor(class_acc).float().mean(),'Precision:',torch.tensor(class_prec).float().mean(),'Recall:',torch.tensor(class_rec).float().mean(),)
        print('Total Correct Samples',np.sum(corr),"out of",len(corr),'=',(np.sum(corr)/len(corr)))
        print("Total Acc:",torch.tensor(acc).mean())
        print("Total Precision:",torch.tensor(prec).mean())
        print("Total Recall:",torch.tensor(rec).mean())

    results = {'percent_correct_samples':(np.sum(corr)/len(corr)),"Total Acc:":float(torch.tensor(acc).mean()),"Total Precision:":float(torch.tensor(prec).mean()),"Total Recall:":float(torch.tensor(rec).mean()),"Class_dict":class_dict}

    # Add total training loss to results 
    results['train_loss'] = running_loss
    print(results)

    # Adding other metrics to results to pass to csv
    results['valid_loss'] = valid_loss
    results['wandb_id'] = experiment.id
    print(experiment.id)
    results['start_time'] = experiment.start_time
    results['train_time'] = duration
    results['stop_epoch'] = stop_epoch


    # Save results to temporary file
    with open(eval_config['pickle_file'],'wb') as fi:
        pickle.dump(results,fi)
    
    if model_config['model_path'] is not None:
        print('Saving model...')
        torch.save(model, model_config['model_path'])
    else:
        print('model_path not provided. Not saving model')
    print('Finished')
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="yaml file with baseline eval settings", type=str)
    args = parser.parse_args()
    train_and_eval(args.config_file)
