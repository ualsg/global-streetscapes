"""
This script contains functions to be imported to raw_download.py, but it can also be run 
on its own to download Mapillary SVIs (metadata only) from the level-14 vector tile associated 
with each input city's location.

Input: a list of city ID(s) - please specify in the variable 'targets' below
Output: one CSV file per input city ID (except where no SVI is available), containing the 
metadata of all downloaded Mapillary and KartaView SVI - please modify the output directory variable 'save_folder' as needed

Note: 
- Please register for a free access token from Mapillary and insert it in the 'access_token' variable below
- If encounter network error, please try running the script again as the API connection is not always stable
"""

import mapillary.interface as mly
import pandas as pd
import geopandas as gp
import os
from pathlib import Path
import datetime

def filter_date(df, start_date, end_date):
    # create a temporary column date from captured_at (milliseconds from Unix epoch)
    df["date"] = pd.to_datetime(df["captured_at"], unit="ms")
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

def get_mly_gdf(city, start_date, end_date):
    """
    Download data from Mapillary and return as a geodataframe.
    """
    cityname = city['city']
    print(f'Downloading Mapillary data for {cityname}...')
    lon = city['lng']
    lat = city['lat']
    try:
        data = mly.get_image_close_to(longitude=lon, latitude=lat)
        dict_data = data.to_dict()
        gdf = gp.GeoDataFrame.from_features(dict_data)
        if not gdf.empty:
            print('Filtering data based on specified time period...')
            gdf = filter_date(gdf, start_date, end_date)
            gdf['city_id'] = [city['id']] * len(gdf)
            gdf['lat'] = gdf.geometry.y
            gdf['lon'] = gdf.geometry.x
            nSeqs = gdf['sequence_id'].nunique()
            print(f'Download complete, collected', nSeqs, 'sequences', len(gdf), 'points')
        return gdf
        # ls_gdf.append(gdf)
    except Exception as e:
        print('network error', e)
        print('No Mapillary data found for', city['city'])


def save_csv(gdf, city, save_folder):
    """
    Save the geodataframe into a csv.
    """
    filename = city['city_ascii'].replace(
        " ", "-") + '_' + str(city['id']) + '.csv'
    dst_path = os.path.join(save_folder, filename)
    pd.DataFrame(gdf.drop(columns='geometry')).to_csv(dst_path, index=False)
    print('Downloaded Mapillary points for', city['city'])


def download_mly_csv(city, save_folder, start_date, end_date):
    gdf = get_mly_gdf(city, start_date, end_date)
    save_csv(gdf, city, save_folder)


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

    # for each of your chosen cities, find its ID from data/worldcities.csv.
    # remember to check the country information to make sure it's the city you want, as different cities can share the same name, e.g. 'San Francisco'.
    # the below city ids correspond to 'Singapore', 'Stuttgart'.
    targets = [1702341327, 1276171358] # please modify as needed

    start_date = '2024-04-01' # start date to download data - please modify as needed (start_date=None indicates download from the earliest available image)
    end_date = None # end date to download data - please modify as needed (end_date=None indicates download until the latest available image)

    # directory to save the downloaded data
    save_folder = Path(__file__).parent / 'sample_output/mly' # please modify as needed
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
        download_mly_csv(city, save_folder, start_date, end_date)
        print('Now:', index, total-len(already_id), 'already:', len(already_id))

    end_size = len([entry for entry in os.listdir(save_folder)
                   if os.path.isfile(os.path.join(save_folder, entry))])
    increase = end_size - start_size
    print('Number of cities with data:', increase, '/', total-len(already_id))
    print('Done')
