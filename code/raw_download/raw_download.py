"""
This script finds the level-14 vector tile associated with each input city's location, downloads and merges all available SVIs (metadata only) 
that fall within this tile, from both Mapillary and KartaView.

This script imports necessary functions from download_kv_points.py and download_mly_points.py, so please keep these three files in the same folder.

If you mean to run the script to update or expand the dataset (i.e. download new data that is not already provided in the dataset), 
please set the 'reproduce' variable to False, and the script would generate and assign a new UUID (Universally unique identifier) 
to each new image downloaded.
If you mean to reproduce the dataset (i.e. download data that is already provided in the dataset), please set the 'reproduce' variable to True, 
and no UUID will be generated as any new UUID generated would be different from the existing ones. But you can easily match the reproduced data 
with the corresponding UUID through a table join based on the their 'source' (Mapillary or KartaView) and their original ID given by their source ('orig_id'). 

Input: a list of city ID(s) - please specify in the variable 'targets' below
Output: one CSV file per input city ID (except where no SVI is available), containing the 
metadata of all downloaded Mapillary and KartaView SVI - please modify the output directory variable 'save_folder' as needed

Note: 
- Please register for a free access token from Mapillary and insert it in the 'access_token' variable below
- If encounter network error, please try running the script again as the API connection is not always stable
"""

import pandas as pd
import os
from pathlib import Path
import uuid
import mapillary.interface as mly
import download_mly_points
import download_kv_points


def download_df(city, zoom, start_date, end_date):
    """
    Download data from both Mapillary and KartaView and merge them into a dataframe.
    """
    try:
        mly_df = download_mly_points.get_mly_gdf(city, start_date, end_date)
        if mly_df.empty:
            print('No images from Mapillary')
            kv_df = download_kv_points.download_kv_df(city, zoom, start_date, end_date)
            if kv_df.empty:
                print('No images from KartaView')
            else:
                kv_df = kv_df.add_prefix('kv_')
                kv_df['source'] = 'KartaView'
                kv_df = kv_df.rename(columns={'kv_heading': 'heading', 'kv_id': 'orig_id',
                            'kv_city_id': 'city_id', 'kv_lat': 'lat', 'kv_lon': 'lon', 'kv_lng': 'lon'})
                if reproduce == False:
                    kv_df['uuid'] = kv_df.apply(lambda row: str(uuid.uuid4()), axis=1)
                return kv_df
        else:
            kv_df = download_kv_points.download_kv_df(city, zoom, start_date, end_date)
            if kv_df.empty:
                print('No images from KartaView')
                mly_df = mly_df.drop(columns='geometry')
                mly_df = mly_df.add_prefix('mly_')
                mly_df['source'] = 'Mapillary'
                mly_df = mly_df.rename(columns={'mly_compass_angle': 'heading', 'mly_id': 'orig_id',
                            'mly_city_id': 'city_id', 'mly_lat': 'lat', 'mly_lon': 'lon'})
                if reproduce == False:
                    mly_df['uuid'] = mly_df.apply(lambda row: str(uuid.uuid4()), axis=1)
                return mly_df
            else:
                mly_df = mly_df.drop(columns='geometry')
                mly_df = mly_df.add_prefix('mly_')
                mly_df['source'] = 'Mapillary'
                kv_df = kv_df.add_prefix('kv_')
                kv_df['source'] = 'KartaView'
                mly_df = mly_df.rename(columns={'mly_compass_angle': 'heading', 'mly_id': 'orig_id',
                'mly_city_id': 'city_id', 'mly_lat': 'lat', 'mly_lon': 'lon'})
                kv_df = kv_df.rename(columns={'kv_heading': 'heading', 'kv_id': 'orig_id',
                'kv_city_id': 'city_id', 'kv_lat': 'lat', 'kv_lon': 'lon'})
                df = pd.concat([mly_df, kv_df]).reset_index(drop=True)
                if reproduce == False:
                    df['uuid'] = df.apply(lambda row: str(uuid.uuid4()), axis=1)
                return df
    except Exception as e:
        print(e)


def save_csv(df, city, save_folder):
    """
    Save the merged dataframe into a csv.
    """
    try:
        filename = city['city_ascii'].replace(
            " ", "-") + '_' + str(city['id']) + '.csv'
        dst_path = os.path.join(save_folder, filename)
        df.to_csv(dst_path, index=False)
        print('Downloaded SVI for',
            city['city'], ':', len(df), 'points')
    except AttributeError:
        print('No images found from both sources')


def download_pts_csv(city, save_folder, start_date, end_date, zoom):
    df = download_df(city, zoom, start_date, end_date)
    save_csv(df, city, save_folder)


def check_id(save_folder):
    """
    Check the save directory for any cities that have already been downloaded to skip download for them.
    """
    ids = set()
    for name in os.listdir(save_folder):
        if name != '.DS_Store':
            ids.add(name.split('_')[1].split('.')[0])
    return ids


if __name__ == '__main__':

    access_token = 'INSERT-YOUR-TOKEN-HERE'  # insert your access token here. access token can be registered on Mapillary for free.
    mly.set_access_token(access_token)

    # set this variable to True if you wish to reproduce the dataset - no UUID will be generated for the data downloaded, please match the downloaded data with existing data based on the 'source' and 'orig_id' attributes to obtain their UUIDs.
    # if you wish to expand (download data for new cities) or update (download new data for existing cities) the dataset, set this variable to False, and UUIDs will be generated for the data downloaded.
    reproduce = False
    if reproduce:
        print(f'Reproduce is set to {reproduce}. No uuids will be generated.')

    # for each of your chosen cities, find its ID from data/worldcities.csv.
    # remember to check the country information to make sure it's the city you want, as different cities can share the same name, e.g. 'San Francisco'.
    # the below city ids correspond to 'Singapore', 'Stuttgart'.
    targets = [1702341327, 1276171358] # please modify as needed
    
    start_date = '2024-04-01' # start date (format must be: 'YYYY-MM-DD') to download data - please modify as needed (start_date=None indicates download from the earliest available image)
    end_date = None # end date (format must be: 'YYYY-MM-DD') to download data - please modify as needed (end_date=None indicates download until the latest available image)

    # directory to save the downloaded data
    save_folder = Path(__file__).parent / 'sample_output/reproduce_false' # please modify as needed
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    # import the simplemaps worldcities database to get city centre for data download
    wc = pd.read_csv(Path(__file__).parent / 'data/worldcities.csv') # please modify as needed
    
    already_id = check_id(save_folder)
    total = len(targets)
    index = 0
    start_size = len([entry for entry in os.listdir(
        save_folder) if os.path.isfile(os.path.join(save_folder, entry))])

    cities = wc[wc['id'].isin(targets)]

    for _, city in cities.iterrows():

        if str(city['id']) in already_id:
            continue

        index += 1
        print('Downloading data for', city['city'])
        download_pts_csv(city, save_folder, start_date, end_date, zoom=14) # other zoom levels are not supported by Mapillary SDK
        print('Now:', index, total-len(already_id), 'already:', len(already_id))

    end_size = len([entry for entry in os.listdir(save_folder)
                   if os.path.isfile(os.path.join(save_folder, entry))])
    increase = end_size - start_size
    print('Number of cities with data:', increase, '/', total-len(already_id))
    print('Done')