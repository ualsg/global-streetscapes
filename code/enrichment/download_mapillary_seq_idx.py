import pandas as pd
import time
import random
import requests
import threading
import os
from pathlib import Path

access_token = 'INSERT-YOUR-TOKEN-HERE'  # insert your access token here. access token can be registered on Mapillary for free.

def get_seq_indexes(seq_id, save_folder):
    
    random_t = random.randint(1,10)/10
    time.sleep(random_t)
    
    url = f'https://graph.mapillary.com/image_ids?sequence_id={seq_id}&access_token={access_token}'
    
    r = requests.get(url, timeout=None)
    
    timeout_count = 0
    
    # if the request failed, keep trying until success (status code = 200)
    while r.status_code != 200:
        timeout_count += 1 # update the number of timeouts
        print(f'<Timeout> timeout count: {timeout_count}, url: {url}') # print timeout information
        r = requests.get(url, timeout=None) # try again
    
    seq = r.json() # when request is succesful, get a JSON format of the response
    df = pd.DataFrame.from_dict(seq['data'])
    df['id'] = df['id'].astype(int)
    df = df.reset_index().rename(columns={'index': 'sequenceIndex'})
    filename = str(seq_id) + '.csv'
    save_path = os.path.join(save_folder, filename)
    df.to_csv(save_path)
    #ls_df.append(df)

def check_id(save_folder):
    ids = set()
    for name in os.listdir(save_folder):
        if name != '.DS_Store':
            ids.add(name.split('.')[0])
    return ids

if __name__ == '__main__':
    
    df = pd.read_csv('../raw_download/sample_output/points.csv')
    save_folder = './sample_data/mly_seqs'

    Path(save_folder).mkdir(parents=True, exist_ok=True)
    
    df = df[df['source'] == 'Mapillary']
    seqs = df['mly_sequence_id'].unique().tolist()

    alr_seqs = check_id(save_folder)
    threads = []
    num_thread = 100
    index = 0

    for seq in seqs:

        if seq in alr_seqs:
            continue
        
        index += 1
        
        if index % num_thread == 0:
            print('Now:', index, len(seqs), 'already:', len(alr_seqs))
            t = threading.Thread(target=get_seq_indexes, args=(seq, save_folder,))
            threads.append(t)
            for t in threads:
                t.Daemon = True
                t.start()
            t.join()
            time.sleep(0.1)
            threads = []
        else:
            t = threading.Thread(target=get_seq_indexes, args=(seq, save_folder,))
            threads.append(t)

    for t in threads:
        t.Daemon = True
        t.start()
    t.join()

    size = len([entry for entry in os.listdir(save_folder) if os.path.isfile(os.path.join(save_folder, entry))])
    print('Number of sequences obtained:', size, '/', len(seqs))
    print('Done')