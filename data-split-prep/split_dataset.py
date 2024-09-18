import numpy as np
import pandas as pd
import sys

#sys.path.insert(0,'/home/lmeyers/beeid_clean_luke/CREATE_DATASET/')

from my_bees_reid_dataset import *
from datetime import datetime
import argparse
import yaml
import os


def split_dataset_function(config_file):
    try:
        with open(config_file) as f:
            config = yaml.safe_load(f)
        #print(config)
        dataset = config['dataset']
        csv_file = config['csv_file']
        test_seed = config['test_seed']
        valid_seed = config['valid_seed']
        train_test_percent = config['train_test_percent']
        train_valid_percent = config['train_valid_percent']
        group_by = config['group_by']
        label_col = config['label_col']
        sort_col = config['sort_col']
        verbose = config['verbose']
    except Exception as e:
        print('ERROR - unable to open config file. Terminating.')
        print('Exception msg:',e)
        return -1
    
    prefixes = [dataset+value for value in ['_train', '_valid', '_test']]
    infix = csv_file[len(dataset):-4]
    split_type = ['_labelDependent_trackIndependent', '_labelDependent_trackIndependent_timeSorted', 
                  '_labelDependent_trackDependent', '_labelDependent_trackDependent_timeSorted']
    suffix = '.csv'
    
    df = pd.read_csv(csv_file)
    # for splitting by tracks, ID-dependent, track-independent
    np.random.seed(test_seed)
    df_train_, df_test = train_test_split_by_label_groupby_sorted(df, label_col, group_by,sort_col,train_test_percent)
    np.random.seed(valid_seed)
    # train/valid random split per ID
    df_train, df_valid = train_validation_split(df_train_, label_col, train_valid_percent)
    fname = prefixes[0] + infix + split_type[0] + suffix
    df_train.to_csv(fname, index=False)
    fname = prefixes[1] + infix + split_type[0] + suffix
    df_valid.to_csv(fname, index=False)
    fname = prefixes[2] + infix + split_type[0] + suffix
    df_test.to_csv(fname, index=False)
    
    # for splitting by tracks; ID-dependent, track-independent, sorted
    # train/valid split is random, per ID or track
    np.random.seed(test_seed)
    df_train_, df_test = train_test_split_by_label_groupby_sorted(df, label_col, group_by, sort_col, train_test_percent, verbose)
    np.random.seed(valid_seed)
    # train/valid random split per ID
    df_train, df_valid = train_validation_split(df_train_, label_col, train_valid_percent)
    fname = prefixes[0] + infix + split_type[1] + suffix
    df_train.to_csv(fname, index=False)
    fname = prefixes[1] + infix + split_type[1] + suffix
    df_valid.to_csv(fname, index=False)
    fname = prefixes[2] + infix + split_type[1] + suffix
    df_test.to_csv(fname, index=False)
    
    
    # for splitting along tracks; ID- and track-dependent
    df_train_, df_test = train_test_split_label_independent(df, label_col, train_test_percent)
    np.random.seed(valid_seed)
    # train/valid random split per ID
    df_train, df_valid = train_validation_split(df_train_, label_col, train_valid_percent)
    fname = prefixes[0] + infix + split_type[2] + suffix
    df_train.to_csv(fname, index=False)
    fname = prefixes[1] + infix + split_type[2] + suffix
    df_valid.to_csv(fname, index=False)
    fname = prefixes[2] + infix + split_type[2] + suffix
    df_test.to_csv(fname, index=False)
    
    # for splitting along tracks; ID- and track-dependent, sorted
    df_train_, df_test = train_test_split_by_label_groupby_sorted(df, label_col,group_by,sort_col, train_test_percent)
    np.random.seed(valid_seed)
    # train/valid random split per ID
    df_train, df_valid = train_validation_split(df_train_, label_col, train_valid_percent)
    fname = prefixes[0] + infix + split_type[3] + suffix
    df_train.to_csv(fname, index=False)
    fname = prefixes[1] + infix + split_type[3] + suffix
    df_valid.to_csv(fname, index=False)
    fname = prefixes[2] + infix + split_type[3] + suffix
    df_test.to_csv(fname, index=False)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("config_file", help="yaml file with experiment settings", type=str)
    args = parser.parse_args()

    split_dataset_function(args.config_file)