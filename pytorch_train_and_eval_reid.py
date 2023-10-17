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
        experiment = wandb.init(project= train_config["wandb_project_name"],entity=train_config['wandb_entity_name'],resume=True,id=train_config['wandb_run_id'])
    else:
        experiment = wandb.init(project= train_config["wandb_project_name"],entity=train_config['wandb_entity_name'])
    
    
    if verbose:
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
    
    # get dataloaders
    if verbose:
        print('Creating train and valid dataloaders...')
    if model_config['model_class'].startswith('swin3d'):
        train_dataloader = get_track_dataset(data_config, 'train')
        valid_dataloader = get_track_dataset(data_config, 'valid')
    else:
        train_dataloader = get_dataset(data_config, 'train')
        valid_dataloader = get_dataset(data_config, 'valid')

    if verbose:
        try:
            batch = next(iter(train_dataloader))
            print(f'Batch image shape: {batch["image"].shape}')
            print(f'Batch label shape: {batch["label"].shape}')
        except Exception as e:
            print('ERROR - could not print out batch properties')
            print(f'Error msg: {e}')

    # build model
    if verbose:
        print('Building model....')
    model = build_model(model_config)

    
    # load latest saved checkpoint if resuming a failed run
    if resume_training == True: 
        saved = os.listdir(os.path.dirname(model_config['model_path'])+r'/checkpoints/')
        check_array = []
        for f in saved:
            check_array.append(f[:-4])
        check_array = np.array(check_array,dtype=np.int64)
        #most_recent_epoch = np.max(check_array) #find most recent checkpoint
        most_recent_epoch = train_config['checkpoint_to_load'] # make this part of yml
        print(f'Resuming training from saved epoch: {most_recent_epoch}')
        most_recent_model = os.path.dirname(model_config['model_path'])+r'/checkpoints/'+str(most_recent_epoch)+'.pth'
        print(f'Loading saved checkpoint model {most_recent_model}')
        model = torch.load(most_recent_model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'])
    # Initialize optimizer and scheduler
    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.75, verbose=True,min_lr = 1e-5)

    miner = miners.MultiSimilarityMiner()
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
            # get images
            if feature_extractor is None:
                images = data['image'].to(device)
            else:
                # must change image back to PIL format for ViT feature extractor
                images = [transforms.functional.to_pil_image(x) for x in data['image']]
                images = np.concatenate([feature_extractor(x)['pixel_values'] for x in images])
                images = torch.tensor(images, dtype=torch.float).to(device)
            # get labels
            labels = data['label'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            hard_pairs = miner(outputs, labels)
            loss = loss_fn(outputs, labels, hard_pairs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            experiment.log({
                'train loss': loss.item(),
                'epoch': epoch
                #'learning rate' : lr
            })

            if (k+1)%print_k == 0:
                with torch.no_grad():
                    valid_outputs, valid_labels, valid_loss = get_embeddings(model, valid_dataloader, loss_fn, miner, device, feature_extractor)
                    print(f'[{epoch + 1}, {k + 1:5d}] train_loss: {running_loss/print_k:.4f} | val_loss: {valid_loss:.4f}')
                    running_loss=0.0
                    #scheduler.step(valid_loss)
                    #current_lr = optimizer.param_groups[0]['lr']
                    experiment.log({'valid loss': valid_loss, })
                                   # 'learning rate': current_lr})

        if epoch % train_config['save_checkpoint_freq'] == 0 or (epoch-1) == train_config['num_epochs']: 
                if os.path.dirname(model_config['model_path']) is not None:
                    print('Saving checkpoint',epoch)
                    if not os.path.exists(os.path.dirname(model_config['model_path'])+r'/checkpoints/'):
                        os.mkdir(os.path.dirname(model_config['model_path'])+r'/checkpoints/')
                    torch.save(model,(os.path.dirname(model_config['model_path'])+r'/checkpoints/'+str(epoch)+".pth"))

    stop = time.time()
    print(f'Total train time: {(stop-start)/60}min')

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
        reference_dataloader = get_dataset(data_config, 'reference')
        test_dataloader = get_dataset(data_config, 'query')

    reference_embeddings, reference_labels, reference_loss = get_embeddings(model, reference_dataloader, loss_fn, miner, device, feature_extractor)
    test_embeddings, test_labels, test_loss = get_embeddings(model, test_dataloader, loss_fn, miner, device, feature_extractor)

    print(f'Reference (or Train) Loss: {reference_loss:.4f}')
    print(f'Test (or Query) Loss: {test_loss:.4f}')

    results = knn_evaluation(reference_embeddings, reference_labels, test_embeddings, test_labels, 
                            eval_config['n_neighbors'], eval_config['per_class'], eval_config['conf_matrix'])
    
    if model_config['model_path'] is not None:
        print('Saving model...')
        torch.save(model, model_config['model_path'])
    else:
        print('model_path not provided. Not saving model')
    print('Finished')
    wandb.finish()

print("beginning execution")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="yaml file with experiment settings", type=str)
    args = parser.parse_args()
    train_and_eval(args.config_file)
