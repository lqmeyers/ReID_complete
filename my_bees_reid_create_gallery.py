import numpy as np
import pandas as pd
from my_bees_reid_dataset import *
from datetime import datetime
import argparse
import yaml
import os

###################################################################################################
#
# FILE USED TO GENERATE GALLERIES
#
###################################################################################################

def gallery_function(config_file):
    try:
        with open(config_file) as f:
            config = yaml.safe_load(f)
        dataset = config['dataset']
        n_iterations = config['n_iterations']
        n_galleries = config['n_galleries']
        n_distractors = config['n_distractors']
        train_fname = config['train_fname']
        test_fname = config['test_fname']
        label_col = config['label_col']
        fname_col = config['fname_col']
        numpy_seed = config['numpy_seed']
        split_type = config['split_type']
    except Exception as e:
        print('ERROR - unable to open config file. Terminating.')
        print('Exception msg:',e)
        return -1
    df_train = pd.read_csv(train_fname)
    df_test = pd.read_csv(test_fname)
    np.random.seed(numpy_seed)
    df_galleries = generate_gallery_dataframe_v2(df_train, df_test, label_col, fname_col, n_iterations, n_galleries, n_distractors)
    # store dataframe
    gallery_fname = os.path.dirname(train_fname)+r'/'+dataset + '_galleries_' + split_type + '.csv'
    df_galleries.to_csv(gallery_fname, index=False)
    print("saved to",gallery_fname)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="yaml file with experiment settings", type=str)
    args = parser.parse_args()
    gallery_function(args.config_file)
