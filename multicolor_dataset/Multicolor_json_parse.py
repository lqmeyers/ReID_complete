#LUke Meyers
#2023_09_16

#doc to parse the .json metadata file for the multicolor baby bees dataset and create CSV that 
#can be used to create a binary color map from the label #

import json 
import csv 
import numpy as np 
import os 
import pandas as pd 

path = "/home/lmeyers/SLEAP_files/Bee_imgs/baby_bee_imgs/Multicolor/Metadata/baby_bees_ids.json"

with open(path,'r') as p:
    data = json.load(p)

dir = '/home/lmeyers/SLEAP_files/Bee_imgs/baby_bee_imgs/Multicolor/Metadata/'


dir_list = os.listdir(dir)
dir_list.sort()
for d in dir_list:
    if "ID" in d:
        n = d[3:7]
        print(n)

color_dict = {}
color_array = []

for id in data['Multicolor']['IDs']:
    #print(id)
    colors_present = []
    for anno in data['Multicolor']['IDs'][id]:
        print(id)
        if anno['type'] == 'Paint':
            colors_present.append(anno['color'])
            color_array.append(anno['color'])
        color_dict[id]= list(np.unique(colors_present))


color_list = ['dark blue', 'dark red', 'green', 'lavender', 'light blue', 'olive', 'orange', 'pink', 'purple', 'red', 'silver', 'white', 'yellow']

color_map = {}

for key in color_dict:
    color_map[key]=list(np.zeros([13]))
    for c in color_dict[key]:
        color_map[key][color_list.index(c)] = 1

#with open("/home/lmeyers/SLEAP_files/Bee_imgs/baby_bee_imgs/Multicolor/Metadata/multicolor_color_map.json", 'w') as f:
   # json.dump(color_map,f)

print(color_dict)


    