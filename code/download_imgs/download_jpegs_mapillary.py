"""
This script contains functions imported by download_jpegs.py.
It can also be run on its own to download Mapillary images.

Input format requirement: a csv file with each row representing an image to download and
containing minimally three columns to specify its 'uuid' (the uuid assigned to the image),
'source' (whether its source is 'Mapillary' or 'KartaView'), and 'orig_id' (original ID as
given by the source).
"""

import pandas as pd
import os
import urllib
import threading
import mapillary.interface as mly
import time
import random
from pathlib import Path

def download_image_from_url(image_url, dst_path):
    try:
        with urllib.request.urlopen(image_url) as web_file:
            data = web_file.read()
            with open(dst_path, mode='wb') as local_file:
                local_file.write(data)
    except urllib.error.URLError as e:
        print('network error', e)


def get_image_url(image_id):
    '''
    automatically download image for each row in the dataframe and append the image filename to the dataframe
    '''
    try:
        random_t = random.randint(1, 10)/10
        time.sleep(random_t)
        image_url = mly.image_thumbnail(image_id, 2048)
        return image_url
        # print('Successed')
        # return os.path.basename(url_2048)
    except Exception as e:
        print('network error', e)


def download_image(image_id, dst_path):
    image_url = get_image_url(image_id)
    download_image_from_url(image_url, dst_path)


def check_id(image_folder):
    ids = set()
    for name in os.listdir(image_folder):
        if name != '.DS_Store':
            ids.add(name.split('.')[0])
    return ids


if __name__ == '__main__':

    access_token = 'INSERT-YOUR-TOKEN-HERE' # update your mapillary access token
    mly.set_access_token(access_token)

    # Update in_csvPath and out_jpegFolder to suit your needs
    in_csvPath = '../raw_download/sample_output/points.csv' # input csv
    out_jpegFolder = './sample_output/mly' # output folder to store the downloaded images
    Path(out_jpegFolder).mkdir(parents=True, exist_ok=True)

    threads = []
    num_thread = 100
    already_id = check_id(out_jpegFolder)

    data_l = pd.read_csv(in_csvPath)
    data_l = data_l[data_l['source']=='Mapillary']

    index = 0

    for _, values in data_l.iterrows():
        image_id = values['orig_id']
        if str(image_id) in already_id:
            continue

        uuid = values['uuid']
        dst_path = os.path.join(out_jpegFolder, uuid + '.jpeg')
        index += 1
        if index % num_thread == 0:
            print('Now:', index, len(data_l)-len(already_id),
                  'already:', index + len(already_id))
            t = threading.Thread(target=download_image,
                                 args=(image_id, dst_path,))
            threads.append(t)
            for t in threads:
                t.setDaemon(True)
                t.start()
            t.join()
            time.sleep(0.1)
            threads = []
        else:
            t = threading.Thread(target=download_image,
                                 args=(image_id, dst_path,))
            threads.append(t)

    for t in threads:
        t.setDaemon(True)
        t.start()
    t.join()
