"""
This script contains functions to be imported to raw_download.py, but it can also be run 
on its own to download KartaView SVIs (metadata only) from the level-14 vector tile associated 
with each input city's location.

Input: a list of city ID(s) - please specify in the variable 'targets' below
Output: one CSV file per input city ID (except where no SVI is available), containing the 
metadata of all downloaded Mapillary and KartaView SVI - please modify the output directory variable 'save_folder' as needed

Note: if encounter network error, please try running the script again, as the API connection is not always stable
"""

import requests
import mpmath as mp
import math
import geopandas as gp
import pandas as pd
import os
from pathlib import Path
import datetime


def get_tile(lat_deg, lon_deg, zoom):
    """
    Obtain the relevant vector tile identified by (z, x, y) using latitude, longitude, and zoom level as input
    """
    lat_rad = mp.radians(lat_deg)
    n = 2 ** zoom
    xtile = n * ((lon_deg + 180) / 360)
    ytile = n * (1 - (mp.log(mp.tan(lat_rad) + mp.sec(lat_rad)) / mp.pi)) / 2
    return (zoom, math.floor(xtile), math.floor(ytile))


def tile2lon(z, x, y):
    """
    Get the longitude of the vector tile
    """
    return x / 2**z * 360 - 180


def tile2lat(z, x, y):
    """
    Get the latitude of the vector tile
    """
    n = mp.pi - 2 * mp.pi * y / 2**z
    return float((180 / mp.pi) * (mp.atan(0.5 * (mp.exp(n) - mp.exp(-n)))))


def tile_bbox(z, x, y):
    """
    Get the bounding box in west(longitude), north(latitude), east(longitude), and south(latitude) of a vector tile
    """
    w = tile2lon(z, x, y)
    n = tile2lat(z, x, y)
    e = tile2lon(z, x+1, y)
    s = tile2lat(z, x, y+1)
    return [w, n, e, s]


def get_bbox(lat, lon, zoom):
    (z, x, y) = get_tile(lat, lon, zoom)
    [w, n, e, s] = tile_bbox(z, x, y)
    return [w, n, e, s]


def get_data_from_url(url):
    """
    Download data from url and return in json
    """
    try:
        timeout_count = 0
        r = requests.get(url, timeout=None)
        while r.status_code != 200:
            timeout_count += 1  # update the number of timeouts
            # print timeout information
            print(f'timeout count: {timeout_count}, url: {url}')
            r = requests.get(url, timeout=None)  # try again

        if r.json()['status']['apiCode'] == 600:
            data = r.json()['result']['data']  # get a JSON format of the response
            return data
        else:
            print(f'===> empty result from <{url}>')
    except Exception as e:
        print('network error', e)


def data_to_dataframe(data):
    """
    Convert downloaded data from json to dataframe
    """
    ls_fields = list(data[0].keys())
    d = {field: [] for field in ls_fields}
    for image in data:
        for field in ls_fields:
            d[field].append(image[field])
    df = pd.DataFrame.from_dict(d)
    return df


def check_id(save_folder):
    """
    Check the save directory for any cities that have already been downloaded to skip download for them.
    """
    ids = set()
    for name in os.listdir(save_folder):
        if name != '.DS_Store':
            ids.add(name.split('_')[1].split('.')[0])
    return ids


def download_points_for_sequence(seq, ls, bbox):
    """
    Download the metadata for all SVI points within a sequence, given a sequence ID.
    Crop the downloaded points with a bounding box.
    Store the cropped data in a list.
    """
    sequenceId = seq['id']
    url = f"https://api.openstreetcam.org/2.0/sequence/{sequenceId}/photos?itemsPerPage=10000&join=user"
    # print(f'===> retrieving points from url... <URL: {url}>')
    data = get_data_from_url(url)
    if data:
        # print('===> converting data...')
        df = data_to_dataframe(data)
        gdf = gp.GeoDataFrame(
            df, geometry=gp.points_from_xy(df.lng, df.lat)
        )
        w, n, e, s = bbox[0], bbox[1], bbox[2], bbox[3]
        gdf2 = gdf.cx[w:e, s:n]
        # print('===> saving data...')
        df = gdf2.drop(columns=['geometry'])
        ls.append(df)
        if (len(ls) != 0 and len(ls) % 500 == 0):
            df_pts = pd.concat(ls).reset_index(drop=True)
            nSeqs = df_pts['sequenceId'].nunique()
            print('===> Collected', nSeqs, 'sequences', len(df_pts), 'points')
            ls = []
            ls.append(df_pts)
        # print('===> download complete,', len(df), 'sequences collected')
    else:
        print('===> Moving to next sequence...')


def download_sequences_for_city(lat, lng, zoom):
    """
    For each city identified by a pair of latitude and longitude, download all SVI sequences
    from the associated vector tile at a specified zoom level (e.g. 14).
    Return downloaded sequences as a dataframe.
    """
    [w, n, e, s] = get_bbox(lat, lng, zoom)
    url = f"https://api.openstreetcam.org/2.0/sequence/?bRight={s},{e}&tLeft={n},{w}&itemsPerPage=10000"
    print(f'===> retrieving sequences from url... <URL: {url}>')
    data = get_data_from_url(url)
    if data:
        # print('===> converting data...')
        df = data_to_dataframe(data)
        print('===>', len(df), 'sequences collected, downloading points...')
        return df
    else:
        df = pd.DataFrame()
        return df

def filter_date(df, start_date, end_date):
    # create a temporary column date from shotDate (%Y-%m-%d %H:%M:%S)
    df["date"] = pd.to_datetime(df["shotDate"], format='%Y-%m-%d %H:%M:%S')
    # check if start_date and end_date are in the correct format with regex. If not, raise error
    if start_date is not None:
        try:
            start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Incorrect start_date format, should be YYYY-MM-DD")
    if end_date is not None:
        try:
            end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Incorrect end_date format, should be YYYY-MM-DD")
    # if start_date is not None, filter out the rows with date < start_date
    df = (
        df[df["date"] >= start_date] if start_date is not None else df
    )
    # if end_date is not None, filter out the rows with date > end_date
    df = df[df["date"] <= end_date] if end_date is not None else df
    # drop the temporary column date
    df = df.drop(columns="date")
    return df


def download_kv_df(city, zoom, start_date, end_date):
    """
    For each city, download all SVI data from the associated vector tile at a specified zoom level (e.g. 14).
    Return downloaded data as a dataframe.
    """
    cityname = city['city']
    print(f'Downloading KartaView data for {cityname}...')
    lat = city['lat'] # obtain city latitude
    lng = city['lng'] # obtain city longitude
    df_seqs = download_sequences_for_city(lat, lng, zoom) # download sequences from the vector tile associated with the city's location at the specified zoom level
    if df_seqs.empty:
        print('No KartaView data found for', city['city'])
    else:
        ls_df = []
        bbox = get_bbox(lat, lng, zoom)
        df_seqs.apply(lambda seq: download_points_for_sequence(
            seq, ls_df, bbox), axis=1) # download SVI points from the sequences
        df_pts = pd.concat(ls_df).reset_index(drop=True)
        if not df_pts.empty:
            print('Filtering data based on specified time period...')
            df_pts = filter_date(df_pts, start_date, end_date)
            df_pts['city_id'] = city['id']
            df_pts = df_pts.drop(columns=['cameraParameters']).rename(columns={'lng': 'lon'}).join(
                df_seqs[
                    ['id',
                        'address',
                        'cameraParameters',
                        'countryCode',
                        'deviceName',
                        'distance',
                        'sequenceType']
                ].set_index('id').rename(columns={'distance': 'distanceSeq'}),
                on='sequenceId',
                how='left'
            ) # append sequence information to each point
        nSeqs = df_pts['sequenceId'].nunique()
        print(f'Download complete, collected', nSeqs, 'sequences', len(df_pts), 'points')
        return df_pts


def save_csv(df_pts, city, save_folder):
    """
    Save downloaded data to a csv
    """
    print('===> download complete,', len(df_pts), 'images collected')
    filename = city['city_ascii'].replace(
        " ", "-") + '_' + str(city['id']) + '.csv'
    dst_path = os.path.join(save_folder, filename)
    df_pts.to_csv(dst_path, index=False)
    print('Downloaded KartaView points for', city['city'])


def download_kv_csv(city, save_folder, zoom, start_date, end_date):
    """
    Download KartaView data for a city's sample area at specified zoom level and save it as csv
    """
    df = download_kv_df(city, zoom, start_date, end_date)
    save_csv(df, city, save_folder)


if __name__ == '__main__':

    # for each of your chosen cities, find its ID from data/worldcities.csv.
    # remember to check the country information to make sure it's the city you want, as different cities can share the same name, e.g. 'San Francisco'.
    # the below city ids correspond to 'Singapore', 'Stuttgart'.
    targets = [1702341327, 1276171358] # please modify as needed

    start_date = '2024-04-01' # start date to download data - please modify as needed (start_date=None indicates download from the earliest available image)
    end_date = None # end date to download data - please modify as needed (end_date=None indicates download until the latest available image)

    # directory to save the downloaded data
    save_folder = Path(__file__).parent / 'sample_output/kv' # please modify as needed
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    # import the simplemaps worldcities database to get city centre for data download
    wc = pd.read_csv(Path(__file__).parent / 'data/worldcities.csv') # please modify as needed

    already_id = check_id(save_folder)
    total = len(targets)
    index = 0
    zoom = 14 # please modify as needed (note that this cannot be modified for Mapillary download)
    start_size = len([entry for entry in os.listdir(
        save_folder) if os.path.isfile(os.path.join(save_folder, entry))])

    cities = wc[wc['id'].isin(targets)]

    for _, city in cities.iterrows():

        if str(city['id']) in already_id:
            continue

        index += 1
        download_kv_csv(city, save_folder, zoom, start_date, end_date)
        print('Now:', index, total-len(already_id), 'already:', len(already_id))

    end_size = len([entry for entry in os.listdir(save_folder)
                   if os.path.isfile(os.path.join(save_folder, entry))])
    increase = end_size - start_size
    print('Number of cities with data:', increase, '/', total-len(already_id))
    print('Done')

