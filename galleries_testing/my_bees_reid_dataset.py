import numpy as np
import pandas as pd
import os
import re

# FUNCTIONS THAT WERE USED TO CONSTRUCT THE CSV FILES FOR THE FLOWERPATCH EXPERIMENTS


##################################################################################################
# FUNCTION TO GENERATE VIDEO ID
# EXAMPLE OF INPUT VIDEO NAME FORMAT: f5x2002_06_20.mp4, f6.2x2022_06_20.mp4
# EXAMLPE OF OUTPUT VIDEO ID FORMAT: 2022_06_22_f05.00, 2022_06_20_f06.02
#
# INPUTS
# 1) video_name: str, name of video with .mp4 ending
# 2) digit_len: int, specifies number of zeros for padding video id
#
# OUTPUTS
# 1) video_id: str, video id
#
##################################################################################################
def generate_video_id(video_name, digit_len=2):
    if not video_name.startswith('f') or not video_name.endswith('.mp4'):
        print(f'ERROR - incorrect format for video name: {video_name}')
        return -1
    video_id = None
    date = re.findall(r'[\d]{4}\_[\d]{2}\_[\d]{2}', video_name)
    if len(date) == 1:
        video_id = date[0]
    #suffix = re.findall(r'f[\d]{0,2}(:?\.[\d]{1,2})?x', video_name)
    suffix = re.search(r'f[\d]{0,2}(:?\.[\d]{0,2})?x', video_name)
    if type(suffix) == re.Match:
        suffix = suffix[0][1:-1].split('.')
        video_id+= '_f'+ '0'*(digit_len - len(suffix[0])) +suffix[0]
        if len(suffix) == 2:
            video_id+= '.' + '0'*(digit_len - len(suffix[1])) + suffix[1]
        else:
            video_id+='.' + '0'*digit_len
    return video_id
##################################################################################################


##################################################################################################
# FUNCTION TO BUILD DATAFRAME
# IMAGE NAMES CONTAIN VIDEO, TRACK, AND FRAME INFORMATION, ZERO-PADDED
# EX: f5x2022_06_20.mp4.track000054.frame008047.jpg
# 
#
# INPUTS
# 1) dirname: str, 
# 2) label_folders: int, specifies number of zeros for padding video id
# 3) ignore_folder: list str, folders to ignore, if needed
# 4) digit_len: int, specifies number of zeros for padding video id
# 5) keep_orig_video: bool, whether to keep original video name
#
# OUTPUTS
# 1) df: pandas dataframe
#
##################################################################################################
def build_dataframe(dirname, label_folders, ignore_folders=[], digit_len=2, keep_orig_video=False):
    labels = []
    fnames = []
    tracks = []
    frames = []
    videos = []
    videos_orig = []
    for folder in label_folders:
        if folder not in ignore_folders:
            label_files = os.listdir(dirname+folder)
            for file in label_files:
                labels.append(folder)
                fnames.append(dirname+folder+'/'+file)
                # get track number
                temp = re.findall(r'track[\d]+', file)
                if len(temp) == 1:
                    tracks.append(int(temp[0][5:]))
                # get frame number
                temp = re.findall(r'frame[\d]+', file)
                if len(temp) == 1:
                    frames.append(int(temp[0][5:]))
                temp = re.findall(r'^[\w\.]+\.mp4', file)
                if len(temp) == 1:
                    videos_orig.append(temp[0])
                    temp = generate_video_id(temp[0], digit_len)
                    videos.append(temp)
    if keep_orig_video:
        df = pd.DataFrame({'filename':fnames, 'label':labels, 'video':videos, 'video_orig':videos_orig, 'track':tracks, 'frame':frames})
        colnames = ['ID', 'paintcode', 'label', 'video', 'video_orig','track', 'frame', 'filename']
    else:
        df = pd.DataFrame({'filename':fnames, 'label':labels, 'video':videos, 'track':tracks, 'frame':frames})
        colnames = ['ID', 'paintcode', 'label', 'video', 'track', 'frame', 'filename']
    # split label into ID and paint code
    df['ID'] = None
    df['paintcode'] = None
    for index, row in df.iterrows():
        results = re.findall(r'ID#[\d]+', row['label'])
        if len(results) == 1:
            sample_id = results[0][3:]
            df.loc[index,('ID')] = sample_id
            temp = len(results[0])
            df.loc[index,('paintcode')] = row['label'][temp:]
    df = df[colnames]
    return df
##################################################################################################


##################################################################################################
# FUNCTION TO GET COLORS FOR A GIVEN LABEL/PAINTCODE
#
# INPUTS
# 1) label: int, ID of given image filename
# 2) color_dict: dictionary, mapping possible strings in label to a unique string per color
# 3) verbose: bool, whether to print out statements
#
# OUTPUTS
# 1) label_color: list str, containing all colors present in image label
#
##################################################################################################
def get_label_color(label, color_dict, verbose=False):
    
    label_color = []
    # find as many colors as are present in label
    for color,value in color_dict.items():
        results = re.findall(color, label.lower())
        if len(results) == 1:
            label_color.append(value)
    # make sure no duplicates in list
    label_color = list(set(label_color))
    if verbose:
        if len(label_color) == 0:
            print(f'\nDid not find a color for {label}')
        else:
            print(f'\nfound color {label_color} for {label}')
    return label_color
##################################################################################################


##################################################################################################
# FUNCTION TO GET LOCATION OF PAINTCODE
# (IF MORE THAN ONE LOCATION AND MORE THAN ONE COLOR PRESENT, ORDER OF LOCATIONS AND COLORS 
# DO NOT NECESSARILY ALIGN WITH EACH OTHER)
# 
# INPUTS
# 1) label: str, ID of a given image filename
# 2) location_dict: dictionary, mapping possible strings in label to a unique string per location
# 3) verbose: bool, whether to print out comments
#
# OUTPUTS
# 1) label_location: list str, containing all locations in image label
#
##################################################################################################
def get_label_location(label, location_dict, verbose=False):
    
    label_location = []
    for location, value in location_dict.items():
        results = re.findall(location, label.lower())
        if len(results) == 1:
            label_location.append(value)
    label_location = list(set(label_location))
    if verbose:
        if len(label_location) == 0:
            print(f'Did not find a location for {label}')
        else:
            print(f'found location {label_location} for {label}')
    return label_location
##################################################################################################



##################################################################################################
# FUNCTION TO SPLIT LABEL ATTRIBUTES
#
# INPUTS
# 1) label_list: list int, containing labels/IDs
# 2) attribute_functions: 
# 3) func_dicts:
# 4) verbose: bool, whether to print out comments
#
# OUTPUTS
# 1) results: 
#
##################################################################################################
def split_label_attributes(label_list, attribute_functions, func_dicts, verbose=False):
    results = {}
    for label in label_list:
        results[label] = {}
        for func_name, attr_func in attribute_functions.items():
            label_result = attr_func(label, func_dicts[func_name],verbose)
            results[label][func_name] = label_result
    return results
##################################################################################################



##################################################################################################
# FUNCTION TO ASSIGN UNIVERSAL TIME TO FILES
# ASSIGNS UNIQUE TIME STAMP TO EACH IMAGE, USING VIDEO ID (WHICH CAN BE SORTED BY TIME)
# AND FRAME NUMBER
#
# INPUTS
# 1) df: pandas dataframe, containing images
# 2) video_col: str, name of column containing video id
# 3) frame_col: str, name of column containing frame number
# 4) first_video_value: int, shift for assigning unique range for unique time stamps between videos
# 5) new_col_name: str, name of column containing uniform time
# 6) verbose: bool, whether to print out comments
#
# OUTPUTS
# 1) df: pandas dataframe, with uniform time
#
##################################################################################################
def assign_universal_time(df, video_col, frame_col, first_video_value=1000000, new_col_name='uniform_time', verbose=False):
    # map each video to its value
    sorted_video_map = {value:(k+1)*first_video_value for k,value in enumerate(sorted(df[video_col].unique()))}
    if verbose:
        print('Sorted video map:')
        for key, value in sorted_video_map.items():
            print(key, value)
    df[new_col_name] = -1
    for index, row in df.iterrows():
        try:
            new_time = sorted_video_map[row[video_col]] + row[frame_col]
            df.loc[index, (new_col_name)] = new_time
        except:
            continue
    if df[df[new_col_name]==-1].shape[0] != 0:
        print('WARNING - some videos not assigned a universal time value')
    return df
##################################################################################################


##################################################################################################
# FUNCTION TO BUILD DATAFRAME, VERSION 2
#
# ASSUMES THAT IMAGES ARE SPLIT INTO FOLDERS, ONE FOR EACH LABEL
# LABELS CONSIST OF ID NUMBER AND PAINTCODE; E.G., ID#14BlueCenterThor
# IMAGE FILENAME CONTAINS VIDEO, TRACK, AND FRAME INFORMATION, ZERO-PADDED; E.G., f5x2022_06_20.mp4.track000054.frame008047.jpg
#
# INPUTS
# 1) dirname: str, path of directory where the label_folders are
# 2) label_folders: list str, name of folders containing images, one per each label
# 3) meta_config: dictionary, contains one dictionary mapping strings to unique color names, and another mapping strings to
#                             unique location names
# 4) ignore_folders: list str, names of folders that should be ignored when image filenames
# 5) digit_le: int, specifies number of zeros for padding video id
# 6) keep_orig_video: bool, whether to include a column containing the original name of video for each image file
# 7) verbose: bool, whether to print out comments
#
# OUTPUTS
# 1) df: pandas dataframe, with one row per image
# 2) df_meta: pandas dataframe, with metadata info on each bee identity
#
##################################################################################################
def build_dataframe_v2(dirname, label_folders,  meta_config, ignore_folders=[], digit_len=2, keep_orig_video=False, verbose=False):
    labels = []
    fnames = []
    tracks = []
    frames = []
    videos = []
    videos_orig = []
    for folder in label_folders:
        if folder not in ignore_folders:
            label_files = os.listdir(dirname+folder)
            for file in label_files:
                labels.append(folder)
                fnames.append(dirname+folder+'/'+file)
                # get track number
                temp = re.findall(r'track[\d]+', file)
                if len(temp) == 1:
                    tracks.append(int(temp[0][5:]))
                else:
                    tracks.append(-1)
                # get frame number
                temp = re.findall(r'frame[\d]+', file)
                if len(temp) == 1:
                    frames.append(int(temp[0][5:]))
                else:
                    frames.append(-1)
                temp = re.findall(r'^[\w\.]+\.mp4', file)
                if len(temp) == 1:
                    videos_orig.append(temp[0])
                    temp = generate_video_id(temp[0], digit_len)
                    videos.append(temp)
                else:
                    videos_orig.append(-1)
                    videos.append(-1)
    # adhoc view of arrays
    if verbose:
        print('Lengths of labels, fnames, tracks, frames, videos, videos_orig')
        print(len(labels), len(fnames), len(tracks), len(frames), len(videos), len(videos_orig))
        print('Number of null tracks, frames, and vidoes')
        print(sum([1 for x in tracks if x==-1]), sum([1 for x in frames if x==-1]), sum([1 for x in videos if x==-1])  )
    if keep_orig_video:
        df = pd.DataFrame({'filename':fnames, 'label':labels, 'video':videos, 'video_orig':videos_orig, 'track':tracks, 'frame':frames})
        colnames = ['ID', 'video', 'video_orig','track', 'frame', 'uniform_time', 'filename']
    else:
        df = pd.DataFrame({'filename':fnames, 'label':labels, 'video':videos, 'track':tracks, 'frame':frames})
        colnames = ['ID', 'video', 'track', 'frame', 'uniform_time', 'filename']
        
    # add uniform time value
    df = assign_universal_time(df, 'video', 'frame', verbose=verbose)

    # split label into ID and paint code
    label_to_ID = {}
    label_to_paintcode = {}
    for label in df.label.unique():
        results = re.findall(r'ID#[\d]+', label)
        if len(results) == 1:
            #label_to_ID[label] = int(results[0][3:])
            label_to_ID[label] = results[0][3:]
            temp = len(results[0])
            label_to_paintcode[label] = label[temp:]
        else:
            if verbose:
                print(f'ERROR - unable to separate ID and paintcode for label {label}')
            label_to_ID[label] = None
            label_to_paintcode[label] = None
    if verbose:
        print('label_to_ID:',label_to_ID)
    # create dataframe to store info about each ID
    label_list = [key for key, value in sorted(label_to_ID.items(), key=lambda item: item[1])]
    print('label_list:',label_list)
    for label in label_list:
        print(label_to_ID[label])
    metadata = [[label_to_ID[label], label, label_to_paintcode[label]] for label in label_list]
    df_meta = pd.DataFrame(metadata)
    df_meta.columns = ['ID', 'label', 'paintcode']
    # get list of colors and locations
    df_meta['colors'] = None
    df_meta['locations'] = None
    for index, row in df_meta.iterrows():
        df_meta.loc[index, ('colors')] = get_label_color(row['label'], meta_config['color_dict'], verbose)
        df_meta.loc[index, ('locations')] = get_label_location(row['label'], meta_config['location_dict'], verbose)
    # change ID to int
    df_meta = df_meta.astype({'ID': 'int32'})
        
    # add ID to df
    df['ID'] = None
    for index, row in df.iterrows():
        df.loc[index, 'ID'] = int(label_to_ID[row['label']])
    # reorder columns
    df = df[colnames]
    return df, df_meta
##################################################################################################



# function for limiting df to (video, track, label) tuples with at least threshold number of images
# this ensures that every track of every ID has a min number of images
##################################################################################################
# FUNCTION TO TRIM DATASET
#
# IT LIMITS DATAFRAME TO (VIDEO, TRACK, LABEL) TUPPLES WITH AT LEAST SOME MINIMUM NUMBER OF IMAGES
#
# INPUTS
# 1) df: pandas dataset
# 2) group_by: list string, list of column names to be used
# 3) threshold: int, minimum number of images required
# 4) verbose: bool, whether to print out comments
#
# OUTPUTS
# 1) df: pandas dataset
#
##################################################################################################
def trim_dataframe(df, group_by, threshold=4, verbose=False):
    index_list = []
    df_group = df.groupby(group_by)
    for group in df_group:
        if group[1].shape[0] >=threshold:
            index_list+=group[1].index.to_list()
    if verbose:
        print(f'Index list length: {len(index_list)}')
    return df.loc[index_list]
##################################################################################################



# FUNCTIONS TO PERFORM TRAIN/VALID/TEST SPLIT

##################################################################################################
# FUNCTION TO PERFORM TRAIN/TEST SPLIT - V1
#
# WAS USED FOR CLOSED SET SETTING (ID DEPENDENT, TRACK INDEPENDENT, TIME SORTED)
# UNIQUE TRACK CONSISTS OF UNIQUE (ID, VIDEO_ID, TRACK) TUPLE
#
# INPUTS
# 1) df: pandas dataframe
# 2) label_col: str, name of column containing label/ID
# 3) group_by: list str, list of columns used to group by
# 4) sort_col: str, name of column used to sort by
# 5) train_percent: float, percent of samples assigned to train set
# 6) verbose: bool, whether to print out comments
#
# OUTPUTS
# 1) df_train: pandas dataframe
# 2) df_test: pandas dataframe
#
##################################################################################################
def train_test_split_by_label_groupby_sorted(df, label_col, group_by, sort_col, train_percent=0.6, verbose=False):
    train_index_list = []
     
    for label in df[label_col].unique():
        group_dict = {}
        df_group = df[df[label_col]==label].groupby(group_by)
        for group in df_group:
            # identify and extract minimum value and corresponding row index
            value_list = group[1][sort_col].values
            init_arg = np.argmin(value_list)
            init_value = value_list[init_arg]
            #init_index = group[1].index.to_list()[init_arg]
            group_dict[group[0]] = init_value
        '''
        for key, value in group_dict.items():
            print(key, value)
        '''
        sorted_group_keys = [key for key, value in sorted(group_dict.items(), key=lambda item: item[1])]
        '''
        for k, value in enumerate(sorted_group_keys):
            print(k,value)
        '''
        # get the number of groups for train
        train_num = int(len(sorted_group_keys)*train_percent)
        #print(f'Train_num: {train_num}')
        # get row indices only from the groups for train
        for group_key in sorted_group_keys[:train_num]:
            train_index_list+=df_group.groups[group_key].to_list()
    df_train = df[df.index.isin(train_index_list)]
    df_test = df[~df.index.isin(train_index_list)]
    return df_train, df_test
##################################################################################################


##################################################################################################
# FUNCTION TO PERFORM TRAIN/VALIDATION SPLIT, PER LABEL
#
# INPUTS
# 1) df: pandas dataframe
# 2) label_col: str, name of column containing label/ID
# 3) train_percent: float, percent of samples assigned to train set
#
# OUTPUTS
# 1) df_train: pandas dataframe
# 2) df_val: pandas dataframe
#
##################################################################################################
def train_validation_split(df, label_col, train_percent=0.8):
    train_index_list = []
    for label in df[label_col].unique():
        index_list = df[df[label_col]==label].index.to_list()
        index_list = list(np.random.permutation(index_list))
        train_num = int(len(index_list)*train_percent)
        train_index_list+= index_list[:train_num]
    df_train = df[df.index.isin(train_index_list)]
    df_val = df[~df.index.isin(train_index_list)]
    return df_train, df_val
##################################################################################################


##################################################################################################
# FUNCTION TO PERFORM TRAIN/TEST SPLIT - V2
#
# WAS USED FOR OPEN SET SETTING (ID INDEPENDENT, TRACK INDEPENDENT)
# 
# INPUTS
# 1) df: pandas dataframe
# 2) label_col: str, name of column containing label/ID
# 3) train_percent: float, percent of samples assigned to train set
#
# OUTPUTS
# 1) df_train: pandas dataframe
# 2) df_test: pandas dataframe
#
##################################################################################################
def train_test_split_label_independent(df, label_col, train_percent=0.8):
    label_list = df[label_col].unique()
    label_list = list(np.random.permutation(label_list))
    train_num = int(len(label_list)*train_percent)
    train_label_list = label_list[:train_num]
    df_train = df[df[label_col].isin(train_label_list)]
    df_test = df[~df[label_col].isin(train_label_list)]
    return df_train, df_test
##################################################################################################


##################################################################################################
# FUNCTION TO GENERATE GALLERIES
#
# INPUTS
# 1) df_train: pandas dataframe, containing images for galleries
# 2) df_test: pandas dataframe, containing images for anchors
# 3) label_col: str, name of column containing label/ID
# 4) fname_col: str, name of column containing filename or path
# 5) N_iter: int, number of iterations for randomly sampling galleries
# 6) N_gall: int, number of galleries to sample per each iteration
# 7) N_distr: int, number of distractors to sample per gallery
#
# OUTPUTS
# 1) df_galleries: pandas dataframe
#
##################################################################################################
def generate_gallery_dataframe_v2(df_train, df_test, label_col, fname_col, N_iter=100, N_gall=100, N_distr = 9):
    
    rows = []

    # for each iteration
    for k in range(N_iter):
        # for each gallery
        for j in range(N_gall):
            # sample an anchor and get its unique filename
            index_A = df_test.sample().index[0]
            id_A = df_test.loc[index_A][label_col]
            file_A = df_test.loc[index_A][fname_col]
            # sample a positive with different filename
            index_P = df_train[(df_train[label_col]==id_A) & (df_train[fname_col]!=file_A)].sample().index[0]
            file_P = df_train.loc[index_P].filename
            # sample distractors
            index_N = df_train[df_train[label_col] != id_A].sample(N_distr, replace=False).index.to_list()
            #store in arrays
            # first store anchor [iteration id, gallery id, 0, filename]
            rows.append([k, j, 0, file_A])
            # now store positive [iteration id, gallery id, 1, filename]
            rows.append([k, j, 1, file_P])
            # now distractors [iteration id, gallery id, >=2, filename]
            for val, index in enumerate(index_N):
                file_N = df_train.loc[index][fname_col]
                rows.append([k, j, 2+val, file_N])
    # create galleries dataframe
    df_galleries = pd.DataFrame(rows)
    df_galleries.columns = ['iteration_id', 'gallery_id', 'image_id', 'filename']
    return df_galleries
##################################################################################################


##################################################################################################
# FUNCTION TO SPLIT TEST SET INTO REFERENCE/QUERY SPLIT (OPEN SET SETTING)
#
# INPUTS
# 1) df: pandas dataframe, containing data
# 2) label_col: str, name of column containing labels/IDs
# 3) group_by: list str, names of columns to perform groupby
# 4) sort_col: str, name of column to sort by
# 5) ref_percent: float, percent of data to be assigned to reference set
# 6) verbose: bool, whether to print out comments
#
# OUTPUTS
# 1) df_ref: pandas dataframe
# 2) df_query: pandas dataframe
#
##################################################################################################
def reference_query_split_by_label_groupby_sorted(df, label_col, group_by, sort_col, ref_percent=0.2, verbose=False):
    ref_index_list = []
    
    for label in df[label_col].unique():
        group_dict = {}
        df_group = df[df[label_col]==label].groupby(group_by)
        for group in df_group:
            # identify and extract minimum value and corresponding row index
            value_list = group[1][sort_col].values
            init_arg = np.argmin(value_list)
            init_value = value_list[init_arg]
            group_dict[group[0]] = init_value
        
        sorted_group_keys = [key for key, value in sorted(group_dict.items(), key=lambda item: item[1])]
        
        # get the number of groups for train
        ref_num = max(1, int(len(sorted_group_keys)*ref_percent)) # make sure at least one track is included
        # get row indices only from the groups for train
        for group_key in sorted_group_keys[:ref_num]:
            ref_index_list+=df_group.groups[group_key].to_list()
    df_ref = df[df.index.isin(ref_index_list)]
    df_query = df[~df.index.isin(ref_index_list)]
    return df_ref, df_query
##################################################################################################

