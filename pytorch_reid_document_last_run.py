import yaml
import os
from glob import glob
import os
import pandas as pd
import json
import pickle
import numpy as np
import sys 


def write_doc_line(config_file):
    '''adds a line of run summary to results tracking document for individual runs'''

    dir = '/home/gsantiago/summer_bee_data/closed_sets_4_ids_all_colors_once_batch2'
    
    #updating datafiles
    train_file = config['data_settings']['datafiles']['train']
    reference_file = config['data_settings']['datafiles']['reference']


    #config['data_settings']['datafiles']['train']=train_csv
    test_file = config['data_settings']['datafiles']['test']
    valid_file = config['data_settings']['datafiles']['valid'] 
    test_file = config['data_settings']['datafiles']['query']

    #make new wandb project based on dir name

    #open config yaml to update experiment params
    with open('/home/lmeyers/ReID_complete/reid_template.yml', 'r') as fo:
        config = yaml.safe_load(fo)

    #run_num = '0'+str(i) #I've decided this makes it harder
    run_str = os.path.basename(train_file)[36:-4]
    run_dir_name = run_str+'/'
    
    split_parts = run_str.rsplit('_', 1)
    # Check if there is at least one underscore in the string
    if len(split_parts) > 1:
        # Get the substring after the last underscore
        num_images = split_parts[1]
        num_ids = split_parts[-1]
    else:
        # Handle the case where there are no underscores in the string
        num_images = run_str
        
    with open('/home/lmeyers/ReID_complete/results.pkl','rb') as fi:
        results = pickle.load(fi)  
        
    # Write out run summary to results tracking document
    results_df = pd.read_csv(config['eval_settings']['results_file'])
    results_df.loc[len(results_df)] = {'run_str': run_str,
                                        'wandb_id':results['wandb_id'],
                                        'num_ids':num_ids,
                                        'num_images_per_id':num_images,
                                        'total_training_images':len(pd.read_csv(train_file)),
                                        'batch_size':config['data_settings']['batch_size'],
                                        'num_epochs':config['train_settings']['num_epochs'],
                                        'train_loss':results['train_loss'],
                                        'valid_loss':results['valid_loss'],
                                        '1NN':results['1NN_acc'],
                                        '3NN':results['3NN_acc'],
                                        'training_file':train_file,
                                        'reference_file':reference_file,
                                        'query_file':test_file,
                                        'start_time':results['start_time'],
                                        'train_time':results['train_time'],
                                        'stop_epoch':results['stop_epoch']}
    results_df.to_csv(config['eval_settings']['results_file'],index=False)
    print("written to:",config['eval_settings']['results_file'])


    
cf = sys.argv[1]
sys.stdout = open(1,'w')

write_doc_line(cf)

        
    
    