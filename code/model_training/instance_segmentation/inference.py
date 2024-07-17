# MIT License 
# Author: winstonyym

import requests
import json
import torch
import glob
import os
import numpy as np
import shutil
import argparse
import logging
import pandas as pd
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import warnings
import time
from tqdm import tqdm
from collections import Counter
from PIL import ImageFile
from datetime import datetime
from pathlib import Path

ImageFile.LOAD_TRUNCATED_IMAGES = True

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 65
CLS_DICT = {
     'index': 'uuid',
     '0': 'Bird',
 '1': 'Ground-Animal',
 '2': 'Curb',
 '3': 'Fence',
 '4': 'Guard-Rail',
 '5': 'Barrier',
 '6': 'Wall',
 '7': 'Bike-Lane',
 '8': 'Crosswalk---Plain',
 '9': 'Curb-Cut',
 '10': 'Parking',
 '11': 'Pedestrian-Area',
 '12': 'Rail-Track',
 '13': 'Road',
 '14': 'Service-Lane',
 '15': 'Sidewalk',
 '16': 'Bridge',
 '17': 'Building',
 '18': 'Tunnel',
 '19': 'Person',
 '20': 'Bicyclist',
 '21': 'Motorcyclist',
 '22': 'Other-Rider',
 '23': 'Lane-Marking---Crosswalk',
 '24': 'Lane-Marking---General',
 '25': 'Mountain',
 '26': 'Sand',
 '27': 'Sky',
 '28': 'Snow',
 '29': 'Terrain',
 '30': 'Vegetation',
 '31': 'Water',
 '32': 'Banner',
 '33': 'Bench',
 '34': 'Bike-Rack',
 '35': 'Billboard',
 '36': 'Catch-Basin',
 '37': 'CCTV-Camera',
 '38': 'Fire-Hydrant',
 '39': 'Junction-Box',
 '40': 'Mailbox',
 '41': 'Manhole',
 '42': 'Phone-Booth',
 '43': 'Pothole',
 '44': 'Street-Light',
 '45': 'Pole',
 '46': 'Traffic-Sign-Frame',
 '47': 'Utility-Pole',
 '48': 'Traffic-Light',
 '49': 'Traffic-Sign-(Back)',
 '50': 'Traffic-Sign-(Front)',
 '51': 'Trash-Can',
 '52': 'Bicycle',
 '53': 'Boat',
 '54': 'Bus',
 '55': 'Car',
 '56': 'Caravan',
 '57': 'Motorcycle',
 '58': 'On-Rails',
 '59': 'Other-Vehicle',
 '60': 'Trailer',
 '61': 'Truck',
 '62': 'Wheeled-Slow',
 '63': 'Car-Mount',
 '64': 'Ego-Vehicle'}


def addInstance(output_max):
    list_unique, list_counts = torch.unique(output_max[0]['segmentation'].int(), return_counts=True)

    if -1 in list_unique:
        list_unique = list_unique[1:]
        list_counts = list_counts[1:]

    total = torch.sum(list_counts).item()

    matching_dict = {}
    for i, k in zip(range(len(output_max[0]['segments_info'])), output_max[0]['segments_info']):
        matching_dict[i] = int(k['label_id'])

    set_dictionary = {}
    for i in range(NUM_CLASSES):
        set_dictionary[str(i)] = 0

    for i, k in zip(list_unique, list_counts):
        set_dictionary[str(matching_dict[i.item()])] += k.item()
        
    set_dictionary['Total'] = total

    return set_dictionary

def addInstanceCounts(output_max):
    
    instance_dictionary = {}
    for i in range(NUM_CLASSES):
        instance_dictionary[str(i)] = 0
    
    # for each segment, draw its legend
    for segment in output_max[0]['segments_info']:
        segment_id = segment['id']
        segment_label_id = str(segment['label_id'])
        instance_dictionary[segment_label_id] += 1

    return instance_dictionary

def check_id(out_csvPath):
    ids = set()
    if os.path.exists(out_csvPath):
        df = pd.read_csv(out_csvPath)
        ls_id = df['uuid'].tolist()
        ids.update(ls_id)
    return ids

# Load Mask2Former
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-panoptic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-panoptic")
model = model.to(device)

# Configure logger (uncomment this section if you wish to log images that failed to run)
# current_time = datetime.now().strftime("%Y%m%d_%H%M")
# logging.basicConfig(filename=f'{current_time}.log', format='%(asctime)s %(message)s', filemode='w') 
# logger=logging.getLogger() 
# logger.setLevel(logging.INFO) 

        
if __name__ == "__main__":

    # folder to store the output CSVs
    out_Folder = "./sample_output"     
    Path(out_Folder).mkdir(parents=True, exist_ok=True)
    # input file path
    in_Path = "../../download_imgs/sample_output/all/img_paths.csv" 
    # the column in input csv that contains the paths to all images
    path_field = 'path'     

    threshold = 0.25

    #logger.info(f"Saving CSV checkpoint for input: {in_file}") # uncomment this if you wish to log images that failed to run
    out_dict = {
        'segmentation': out_Folder + "/" + 'segmentation' + ".csv",
        'instance': out_Folder + "/" + 'instances' + ".csv"
    }
    cols = list(CLS_DICT.values())
    for key, out_csvPath in out_dict.items():
        if os.path.exists(out_csvPath) == False:
            if key == 'segmentation':
                cols = cols + ['Total']
            df = pd.DataFrame(columns=cols)
            df.to_csv(out_csvPath, index=False)
            cols = list(CLS_DICT.values())
    input_df = pd.read_csv(in_Path)
    alr_ids = check_id(out_dict['instance'])
    input_df = input_df[~input_df['uuid'].isin(alr_ids)].reset_index(drop=True)
    start = 0
    results_dict = {key:{} for key in out_dict.keys()}
    for i, img in tqdm(input_df.iterrows(),total=len(input_df)):
        img_path = img[path_field]
        uuid = os.path.basename(img_path).split('.')[0]
        try:
            with Image.open(img_path) as image:
                inputs = processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    pixel_values = inputs['pixel_values'].to(device)
                    pixel_mask = inputs['pixel_mask'].to(device)
                    outputs = model(pixel_values = pixel_values, pixel_mask = pixel_mask)
                out = processor.post_process_instance_segmentation(outputs, target_sizes=[image.size[::-1]], threshold=threshold)
                results_dict['segmentation'][uuid] = addInstance(out)
                results_dict['instance'][uuid] = addInstanceCounts(out)
        except Exception as e:
            print(e)
            print(f"Failed for: {uuid}")
            #logger.info(f"Failed for {i}:{uuid}") # uncomment this if you wish to log images that failed to run

        if i != 0 and i % 100 == 0:
            for key, out_csvPath in out_dict.items():
                df = pd.DataFrame.from_dict(results_dict[key], orient='index').reset_index()
                df = df.rename(mapper = CLS_DICT, axis=1)
                if key == 'segmentation':
                    cols = cols + ['Total']
                df = df[cols]
                df.to_csv(out_csvPath, mode='a', header=False, index=False)
                results_dict[key] = {}
                cols = list(CLS_DICT.values())
            #logger.info(f"Segmented images {start}:{i}") # uncomment this if you wish to log images that have been successfully run
            start = i

        i += 1
    
    try:
        for key, out_csvPath in out_dict.items():
            df = pd.DataFrame.from_dict(results_dict[key], orient='index').reset_index()
            df = df.rename(mapper = CLS_DICT, axis=1)
            if key == 'segmentation':
                cols = cols + ['Total']
            df = df[cols]
            df.to_csv(out_csvPath, mode='a', header=False, index=False)
            results_dict[key] = {}
            cols = list(CLS_DICT.values())
        # logger.info(f"Segmented images {start}:{len(input_df)}") # uncomment this if you wish to log images that have been successfully run
    except KeyError:
        print('All images have been segmented.')

   
