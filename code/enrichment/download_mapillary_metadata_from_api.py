import pandas as pd
import os
import threading
import time
import requests
from tqdm import tqdm
from pandas import json_normalize
from pathlib import Path

access_token = 'INSERT-YOUR-TOKEN-HERE'  # insert your access token here. access token can be registered on Mapillary for free.

def check_id(image_folder):
    ids = set()
    print('Checking all subfolders for existing images...')
    for subdir, dirs, files in os.walk(image_folder):
        # print('Checking', subdir, '...')
        count = 0
        for file in files:
            if file != '.DS_Store':
                ids.add(file.split('.')[0])
                count += 1
        print('Found', count, 'images in', subdir)
    return ids

def get_img_metadata(img_id, fields, dst_path):
    try:
        url = f'https://graph.mapillary.com/{img_id}?access_token={access_token}&fields={fields}'
        r = requests.get(url, timeout=None)
        data = r.json()
        data = json_normalize(data)
        df = pd.DataFrame(data)
        df.to_csv(dst_path, index=False)
    except Exception as e:
        print('network error', e)

if __name__ == '__main__':
    
    read_path = '../raw_download/sample_output/points.csv'
    out_mainFolder = './sample_data/mly_additional_metadata'

    Path(out_mainFolder).mkdir(parents=True, exist_ok=True)
    
    fields = 'id,altitude,atomic_scale,camera_parameters,camera_type,\
        computed_altitude,computed_compass_angle,computed_geometry,computed_rotation,\
            exif_orientation,height,merge_cc,mesh,quality_score,sfm_cluster,width,creator,make,model' 
    chunk_size = 10000
    num_thread = 100

    already_id = check_id(out_mainFolder)
    df_imgs = pd.read_csv(read_path)
    df_imgs = df_imgs[df_imgs['source'] == 'Mapillary'].reset_index(drop=True)
    indices = list(range(0, len(df_imgs), chunk_size))
    ls_df = []
    for i in range(len(indices)-1):
        start = indices[i]
        end = indices[i+1]
        df = df_imgs.iloc[start:end]
        ls_df.append(df)
    df = df_imgs.iloc[indices[-1]:]
    ls_df.append(df)
    imgcnt = 0

    for df in tqdm(ls_df):

        start = df.index[0]+1
        end = df.index[-1]+1
        out_subFolder = f"{start}_{end}"
        threads = []

        index = 0

        for _, values in tqdm(df.iterrows()):

            uuid = values['uuid']
            
            if uuid in already_id:
                continue

            index += 1
            imgcnt += 1
            if os.path.exists(os.path.join(out_mainFolder, out_subFolder)) == False:
                os.mkdir(os.path.join(out_mainFolder, out_subFolder))
            dst_path = os.path.join(out_mainFolder, out_subFolder, uuid +'.csv')
            img_id = values['orig_id']

            if index % num_thread == 0:
                print('Now:', imgcnt, len(df_imgs)-len(already_id), 'already:', len(already_id))
                t = threading.Thread(target=get_img_metadata, args=(img_id, fields, dst_path,))
                threads.append(t)
                for t in threads:
                    t.Daemon = True
                    t.start()
                t.join()
                time.sleep(0.1)
                threads = []
            else:
                t = threading.Thread(target=get_img_metadata, args=(img_id, fields, dst_path,))
                threads.append(t)

        print('Now:', imgcnt, len(df_imgs)-len(already_id), 'already:', len(already_id))
        try:
            for t in threads:
                t.Daemon = True
                t.start()
            t.join()
        except Exception as e:
            print('All metadata for this subfolder have been downloaded.')
