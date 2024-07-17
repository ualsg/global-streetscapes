"""
This script contains functions imported by download_jpegs.py.
It can also be run on its own to download KartaView images.

Input format requirement: a csv file with each row representing an image to download and 
containing minimally three columns to specify its 'uuid' (the uuid assigned to the image), 
'source' (whether its source is 'Mapillary' or 'KartaView'), and 'orig_id' (original ID as 
given by the source).
"""

import pandas as pd
import os
import urllib.request
import threading
import time
import random
import requests
from pathlib import Path

def get_image_url(image_id):
    url = f'https://api.openstreetcam.org/2.0/photo/?id={image_id}'
    try:
        r = requests.get(url, timeout=None)
        while r.status_code != 200:
            r = requests.get(url, timeout=None)  # try again
        try:
            # get a JSON format of the response
            data = r.json()['result']['data'][0]
            image_url = data['fileurlProc']
            return image_url
        except Exception as e:
            print('network error', e)
    except urllib.error.URLError as e:
        print('network error', e)


def download_image_from_url(url, dst_path):
    try:
        random_t = random.randint(1, 10)/10
        time.sleep(random_t)
        with urllib.request.urlopen(url) as web_file:
            data = web_file.read()
            with open(dst_path, mode='wb') as local_file:
                local_file.write(data)
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
    # Update in_csvPath and out_jpegFolder to suit your needs
    in_csvPath = '../raw_download/sample_output/points.csv' # input csv
    out_jpegFolder = './sample_output/kv' # output folder to store the downloaded images
    Path(out_jpegFolder).mkdir(parents=True, exist_ok=True)

    threads = []
    num_thread = 100
    already_id = check_id(out_jpegFolder)

    data_l = pd.read_csv(in_csvPath)
    data_l = data_l[data_l['source']=='KartaVIew']

    index = 0

    for _, values in data_l.iterrows():
        image_id = values['orig_id']
        if str(image_id) in already_id:
            continue

        uuid = values['uuid']
        dst_path = os.path.join(out_jpegFolder, str(image_id) + '.jpeg')
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
