import yaml
import os
from glob import glob
import os
import pandas as pd
import json
import pickle
import numpy as np
import wandb
import torch
import sys 

sys.path.insert(1,'/home/lmeyers/ReID_complete/')

from pytorch_data import *
from pytorch_models import *
from pytorch_train_and_eval_color_detect import *

def eval(config_file):
    with open(config_file, 'r') as fo:
        config = yaml.safe_load(fo)
    model_config = config['model_settings'] # settings for model building
    train_config = config['train_settings'] # settings for model training
    data_config = config['data_settings'] # settings for data loading
    eval_config = config['eval_settings'] # settings for evaluation
    torch_seed = config['torch_seed']
    verbose = config['verbose']


    model = torch.load(model_config['model_path'])

    # LOAD TO DEVICE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    if verbose:
        print(f'Device: {device}')
    model.to(device)

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
                class_dict[c]={'acc:':float(torch.tensor(class_acc).mean()),'precision:':float(torch.tensor(class_prec).mean()),'recall:':float(torch.tensor(class_rec).mean())}
                print('Class',c,'TP:',int(TP),'FN:',int(FN),'FP:',int(FP),'TN:',int(TN),'Acc:',torch.tensor(class_acc).float().mean(),'Precision:',torch.tensor(class_prec).float().mean(),'Recall:',torch.tensor(class_rec).float().mean(),)
        print('Total Correct Samples',np.sum(corr),"out of",len(corr),'=',(np.sum(corr)/len(corr)))
        print("Total Acc:",torch.tensor(acc).mean())
        print("Total Precision:",torch.tensor(prec).mean())
        print("Total Recall:",torch.tensor(rec).mean())

    results = {'percent_correct_samples':(np.sum(corr)/len(corr)),"Total Acc:":float(torch.tensor(acc).mean()),"Total Precision:":float(torch.tensor(prec).mean()),"Total Recall:":float(torch.tensor(rec).mean()),"Class_dict":class_dict}

    # Add total training loss to results 
    #results['train_loss'] = running_loss
    #print(results)

    # Adding other metrics to results to pass to csv
    #results['valid_loss'] = valid_loss
    #results['wandb_id'] = experiment.id
    #print(experiment.id)
    #results['start_time'] = experiment.start_time
    #results['train_time'] = duration
    results['stop_epoch'] = os.path.basename(model_config['model_path'][:-4])


    # Save results to temporary file
    with open(eval_config['pickle_file'],'wb') as fi:
        pickle.dump(results,fi)
    
    if model_config['model_path'] is not None:
        print('Saving model...')
        torch.save(model, model_config['model_path'])
    else:
        print('model_path not provided. Not saving model')
    print('Finished')
    #wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="yaml file with baseline eval settings", type=str)
    args = parser.parse_args()
    eval(args.config_file)
