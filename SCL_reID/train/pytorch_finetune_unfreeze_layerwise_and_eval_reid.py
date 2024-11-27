import numpy as np
import pandas as pd
from PIL import Image as Image2
import matplotlib.pyplot as plt
import time
import argparse
import yaml
import pickle
from datetime import datetime
import random
import gc
import os
import wandb
import pickle
import sys 
sys.path.insert(0,"../")


import torch
from transformers import ViTFeatureExtractor, AutoImageProcessor, CLIPFeatureExtractor
from utils.pytorch_data import *
from models.pytorch_models import *
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity

clip_flattener = CLIPFeatureExtractor()
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
def knn_evaluation(train_images, train_labels, test_images, test_labels, n_neighbors, per_class=True, conf_matrix=True, random_state=101):
    # BUILD KNN MODEL AND PREDICT
    np.random.seed(random_state)
    random.seed(random_state)
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
    
    resume_training = train_config["resume_from_saved"]
    if resume_training == True: 
        if train_config['wandb_run_id'] is not None:
            experiment = wandb.init(project= train_config["wandb_project_name"],entity=train_config['wandb_entity_name'],resume=True,id=train_config['wandb_run_id'],dir=train_config['wandb_dir_path'])
        else:
            experiment = wandb.init(project=train_config["wandb_project_name"],entity=train_config['wandb_entity_name'],dir=train_config['wandb_dir_path'])
    else:
        experiment = wandb.init(project=train_config["wandb_project_name"],entity=train_config['wandb_entity_name'],dir=train_config['wandb_dir_path'])


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
    np.random.seed(torch_seed)
    
    # get dataloaders
    if verbose:
        print('Creating train and valid dataloaders...')
    if model_config['model_class'].startswith('swin3d'):
        train_dataloader = get_track_dataset(data_config, 'train')
        valid_dataloader = get_track_dataset(data_config, 'valid')
    else:
        if data_config['sample_valid'] == True:
            train_dataloader, valid_dataloader = get_dataset(data_config, 'train',generate_valid=True) #generate valid automatically
        else:
            train_dataloader = get_dataset(data_config, 'train')
            valid_dataloader = get_dataset(data_config, 'valid')
    
    #init test dataloaders pre-training for mid run eval
    if verbose:
        print('initializing evaluation dataloaders...')
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
        # saved = os.path.dirname(model_config['model_path'])+r'/checkpoints/'
        # #most_recent_epoch = np.max(check_array) #find most recent checkpoint
        most_recent_epoch = train_config['checkpoint_to_load'] # make this part of yml
        print(f'Resuming training from saved epoch: {most_recent_epoch}')
        # most_recent_model = saved+str(most_recent_epoch)+'.pth'
        most_recent_model = "/home/lmeyers/contrastive_learning_new_training/64_ids_batch1_sample_num_max/wandb/run-20241018_093254-5eva7juu/files/64_ids_batch1_sample_num_max.pth"
        print(f'Loading saved model {most_recent_model}')
        model = torch.load(most_recent_model)
        model.train()
    else:
        most_recent_epoch = 0
    
    
    # Training Hyperparams 
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'])
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.85, verbose=True,min_lr = 1e-5)
    #miner = miners.MultiSimilarityMiner()
    miner = miners.TripletMarginMiner(margin=train_config['margin'], type_of_triplets="semihard", distance = CosineSimilarity())
    miner_type = "semihard"
    loss_fn = losses.TripletMarginLoss(train_config['margin'], distance = CosineSimilarity())
    if verbose:
        print('Loss:',loss_fn)

    ################################# TODO containerize this
    # load VIT feature extractor if needed
    if model_config['model_class'].startswith('vit'):
        if verbose:
            print('Getting ViT feature extractor...')
        model_name = 'google/vit-base-patch16-224-in21k'
        feature_extractor = AutoImageProcessor.from_pretrained(model_name)
        #Frozen model first to finetune head
        if verbose:
            print('Freezing Backbone weights...')
        model.vit.requires_grad_(False) #extra underscore works for whole modules 
        model_frozen = True
        all_layer_num = -1*(len(model.vit.encoder.layer)-1)
    
    elif model_config['model_class'].startswith('clip'):
        if verbose:
            print('Getting clip feature extractor...')
        feature_extractor = CLIPFeatureExtractor()
        model.bio_model.requires_grad_(False) #freeze all
        model_frozen = True
    else:
        feature_extractor = None
        model_frozen = False

    ###########################################

    ##########################################################
    # Set device and send to cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f'Found device: {device}')
    
    ######################################
    model.to(device)

    # if resuming training set epoch number
    if resume_training == True:
        epoch_range = range(train_config['num_epochs'])[most_recent_epoch:]
        stop_epoch = 0

    else:
        epoch_range = range(train_config['num_epochs'])
        stop_epoch = 0

    # Initialize early stopping variables
    best_loss = float('inf')
    best_model = model
    valid_loss = 'Null'
    num_epochs_no_improvement = 0
    check_for_early_stopping = train_config['early_stopping']
    consecutive_epochs = train_config['early_stop_consecutive_epochs']
    finetune_epochs = train_config['finetune_epochs']
    mid_train_eval = False
    unfreeze_epoch = None
    stop_early = False
    unfrozen_layer_num = 0
    
    
    # Train the model
    if verbose:
        print('Training model head...')
    print_k = train_config['print_k']

    start = time.time()
    for epoch in epoch_range: 
        running_loss = 0.0
        for k, data in enumerate(train_dataloader):
            # get images
            if feature_extractor is None:
                if verbose & epoch == 0 & k == 0:
                    print("No tokenizer being used, not transformer model")
                images = data['image'].to(device)
            else:
                # must change image back to PIL format for ViT feature extractor
                images = [transforms.functional.to_pil_image(x) for x in data['image']]
                images = np.concatenate([feature_extractor(x)['pixel_values'] for x in images])
                images = torch.tensor(images, dtype=torch.float).to(device)
            # get labels
            labels = data['label'].to(device)

            #zero the parameter gradients
            optimizer.zero_grad()

            #pass through the model 
            outputs = model(images)

            # get semi-hard triplets
            triplet_pairs = miner(outputs, labels)
            #hard_pairs = miner(outputs, labels)

            #calculate loss and step gradients
            loss = loss_fn(outputs, labels, triplet_pairs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            experiment.log({
                'train loss': loss.item(),
                'epoch': epoch,
                #'learning rate' : lr
                'triplet_num': torch.numel(triplet_pairs[0])
            })

#             if (k+1)%print_k == 0:
        train_loss = running_loss/print_k
        
        if (epoch % train_config['save_checkpoint_freq'] == 0 or (epoch+1) == train_config['num_epochs']) and epoch != most_recent_epoch: 
                if os.path.dirname(model_config['model_path']) is not None:
                    print('Saving checkpoint',epoch)
                    if not os.path.exists(os.path.dirname(model_config['model_path'])+r'/checkpoints/'):
                        os.mkdir(os.path.dirname(model_config['model_path'])+r'/checkpoints/')
                    torch.save(model,(os.path.dirname(model_config['model_path'])+r'/checkpoints/'+str(epoch)+".pth"))
                    
        with torch.no_grad():
            valid_outputs, valid_labels, valid_loss = get_embeddings(model, valid_dataloader, loss_fn, miner, device, feature_extractor)
            
            print(f'[{epoch + 1}, {k + 1:5d}] train_loss: {running_loss/print_k:.4f} | val_loss: {valid_loss:.4f}')
            running_loss=0.0
            #scheduler.step(valid_loss)
            current_lr = optimizer.param_groups[0]['lr']
            experiment.log({'valid loss': valid_loss, "learning_rate": current_lr})
            
            #assign loss to eval for early stopping
            if train_config["early_stopping_metric"] == "train_loss":
                metric_loss = train_loss
            else:
                metric_loss = valid_loss
                
            # Check if validation loss has improved
            if metric_loss < best_loss:
                best_loss = valid_loss
                model_epoch = epoch
                best_model = model
                num_epochs_no_improvement = 0
            else:
                num_epochs_no_improvement += 1

            #use early stopping protocol to unfreeze model and finetune more
            if (num_epochs_no_improvement >= consecutive_epochs) and (model_frozen == True):
                if model_config['model_class'].startswith('vit'):
                    model.vit.encoder.layer[-1].requires_grad_(True) #unfreeze last layer
                    model.vit.pooler.requires_grad_(True) # unfreeze pooler first time
                elif model_config['model_class'].startswith('clip'):
                    model.bio_model.requires_grad_(True)
                num_epochs_no_improvement = 0
                best_loss = float('inf') # reset best model
                consecutive_epochs = consecutive_epochs//2 #shorten patience for layerwise
                optimizer.param_groups[0]['lr'] = 1e-4
                unfrozen_layer_num -= 1
                model_frozen = False
                unfreeze_epoch = epoch+1
                mid_train_eval = eval_config['mid_train_eval']
                print(f'Unfreezing pooler and last layer of backbone at {epoch+1} due to no improvement in {train_config["early_stopping_metric"]} for {consecutive_epochs} consecutive epochs')
                print(f'Will continue to train for {finetune_epochs} with learning rate of {optimizer.param_groups[0]["lr"]} .')
                
            #if model has plateau'd and is partly unfozen, unfreeze another layer
            if (num_epochs_no_improvement >= consecutive_epochs) and (all_layer_num != unfrozen_layer_num):
                  if model_config['model_class'].startswith('vit'):
                    unfrozen_layer_num -= 1
                    model.vit.encoder.layer[unfrozen_layer_num].requires_grad_(True) #unfreeze successive layer
                    print(f'Unfreezing layer {unfrozen_layer_num} of backbone at {epoch+1} due to no improvement in {train_config["early_stopping_metric"]} for {consecutive_epochs} consecutive epochs')
                    if unfrozen_layer_num <= (all_layer_num /2): #slow down learning upon unfreezing early layers
                        optimizer.param_groups[0]['lr'] = 1e-5
                    num_epochs_no_improvement = 0 #reset check for early stopping but not best model
                                
            #check if number of finetuning epochs has been exceeded
            if (model_frozen == False):
                if epoch+1 >= ((epoch+1) + finetune_epochs):
                    print(f'Finished {finetune_epochs} finetuning epochs')
                    stop_early = True
            
            # Check if early stopping condition is met
            if check_for_early_stopping == True:
                if (num_epochs_no_improvement >= consecutive_epochs):
                    print(f'Early stopping at epoch {epoch+1} due to no improvement in {train_config["early_stopping_metric"]} for {consecutive_epochs} consecutive epochs')
                    stop_epoch = epoch+1
                    stop_early = True 
            
    #       #eval on test set mid training for efficient testing
            if (mid_train_eval == True) and (epoch%eval_config["eval_every_epoch"] == 0): 
                print(f"evaluating on test set after {epoch + 1} epochs")
                print(f"using model from epoch {model_epoch} with loss: {best_loss}")
                reference_embeddings, reference_labels, reference_loss = get_embeddings(model, reference_dataloader, loss_fn, miner, device, feature_extractor)
                test_embeddings, test_labels, test_loss = get_embeddings(model, test_dataloader, loss_fn, miner, device, feature_extractor)
                print(f'Reference (or Train) Loss: {reference_loss:.4f}')
                print('Reference size:',reference_embeddings.shape)
                print(f'Test (or Query) Loss: {test_loss:.4f}')
                print('Test (or Query) size:',test_embeddings.shape)
                results = knn_evaluation(reference_embeddings, reference_labels, test_embeddings, test_labels, 
                                eval_config['n_neighbors'], False, False)
              
        #Breaks Epoch iteration to stop training early
        # will only be true if checking for early stopping is enabled                     
        if stop_early == True:
            break

    #---- Perform eval with best model---------
    model = best_model
    #print("using latest model instead of best")
    if verbose:
        print(f"using epoch {model_epoch} with loss {best_loss} for eval")
    stop_epoch = epoch+1
    stop = time.time()
    duration = (stop-start)/60
    print(f'Total train time: {duration}min')

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
    
    if verbose:
        print('generating embeddings')

    reference_embeddings, reference_labels, reference_loss = get_embeddings(model, reference_dataloader, loss_fn, miner, device, feature_extractor)
    test_embeddings, test_labels, test_loss = get_embeddings(model, test_dataloader, loss_fn, miner, device, feature_extractor)

    # Convert query/test labels to match referene labels if necessary 
    b1_to_b2 = {10: 74, 11: 75, 15: 79, 14: 78, 12: 76, 16: 80, 13: 77, 9: 73, 18: 82, 19: 83, 23: 87, 22: 86, 20: 84, 24: 88, 21: 85, 17: 81, 50: 114, 51: 115, 55: 119, 54: 118, 52: 116, 56: 120, 53: 117, 49: 113, 42: 106, 43: 107, 47: 111, 46: 110, 44: 108, 48: 112, 45: 109, 41: 105, 26: 90, 27: 91, 31: 95, 30: 94, 28: 92, 32: 96, 29: 93, 25: 89, 58: 122, 59: 123, 63: 127, 62: 126, 60: 124, 64: 128, 61: 125, 57: 121, 34: 98, 35: 99, 39: 103, 38: 102, 36: 100, 40: 104, 37: 101, 33: 97, 2: 66, 3: 67, 7: 71, 6: 70, 68: 68, 8: 72, 5: 69, 1: 65}
    b2_to_b1 = {74: 10, 75: 11, 79: 15, 78: 14, 76: 12, 80: 16, 77: 13, 73: 9, 82: 18, 83: 19, 87: 23, 86: 22, 84: 20, 88: 24, 85: 21, 81: 17, 114: 50, 115: 51, 119: 55, 118: 54, 116: 52, 120: 56, 117: 53, 113: 49, 106: 42, 107: 43, 111: 47, 110: 46, 108: 44, 112: 48, 109: 45, 105: 41, 90: 26, 91: 27, 95: 31, 94: 30, 92: 28, 96: 32, 93: 29, 89: 25, 122: 58, 123: 59, 127: 63, 126: 62, 124: 60, 128: 64, 125: 61, 121: 57, 98: 34, 99: 35, 103: 39, 102: 38, 100: 36, 104: 40, 101: 37, 97: 33, 66: 2, 67: 3, 71: 7, 70: 6, 68: 68, 72: 8, 69: 5, 65: 1}

    reference_data_batch = os.path.dirname(data_config['datafiles']['reference'])[-1:]
    query_data_batch = os.path.dirname(data_config['datafiles']['query'])[-1:]

    if reference_data_batch != query_data_batch and data_config['label_col'] != 'color_num':
        if reference_data_batch > query_data_batch:
            for i in range(len(test_labels)):
                test_labels[i] = b1_to_b2[test_labels[i]]
        else: 
            for i in range(len(test_labels)):
                test_labels[i] = b2_to_b1[test_labels[i]]
    
    print(f'Reference (or Train) Loss: {reference_loss:.4f}')
    print('Reference size:',reference_embeddings.shape)
    print(f'Test (or Query) Loss: {test_loss:.4f}')
    print('Test (or Query) size:',test_embeddings.shape)

    results = knn_evaluation(reference_embeddings, reference_labels, test_embeddings, test_labels, 
                            eval_config['n_neighbors'], eval_config['per_class'], eval_config['conf_matrix'])
    
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
    results['unfreeze_epoch'] = unfreeze_epoch

    # if not os.path.exists(eval_config['pickle_file']):
    #     with open(eval_config['pickle_file'],'ab'):
    #         os.utime(eval_config['pickle_file'], None)

    # Save results to temporary file
    with open(eval_config['pickle_file'],'wb') as fi:
        pickle.dump(results,fi)
        print("Results saved to pickle file")

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
