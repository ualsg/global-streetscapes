import pandas as pd
import geopandas as gp
import os

def split_df(df, ith, ls):
    for i in range(0, len(df), ith):
        if i < (len(df) - ith):
            s = df.iloc[i:i+ith]
            ls.append(s)
        else:
            s = df.iloc[i:]
            ls.append(s)

def append_gadm(i, df, ls_gadm):
    print(i)
    if i > 0:
        df_right = df.sjoin(ls_gadm[i], how='left')
        df_left = append_gadm(i-1, df, ls_gadm)
        cols_to_use = df_right.columns.difference(df_left.columns)
        return pd.merge(df_left, df_right[cols_to_use], left_index=True, right_index=True, how='outer')
    else:
        df_left = df.sjoin(ls_gadm[i], how="left")
        return df_left

def check_id(save_path):
    ids = set()
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
        ls_id = df['uuid'].tolist()
        ids.update(ls_id)
    return ids

if __name__ == '__main__':

    filepath = 'insert the path to your gadm_410-levels.gpkg'
    gadm0 = gp.read_file(filepath, layer='ADM_0')
    gadm1 = gp.read_file(filepath, layer='ADM_1')
    gadm2 = gp.read_file(filepath, layer='ADM_2')
    gadm3 = gp.read_file(filepath, layer='ADM_3')
    gadm4 = gp.read_file(filepath, layer='ADM_4')
    gadm5 = gp.read_file(filepath, layer='ADM_5')
    print('Finished reading all gadm layers')

    cols = ['uuid', 'source', 'orig_id', 'lat', 'lon']
    pts = pd.read_csv('./sample_data/02_metadata_common_attributes.csv')[cols]
    save_path = './sample_data/08_gadm.csv'

    gdf = gp.GeoDataFrame(
        pts, geometry=gp.points_from_xy(pts.lon, pts.lat), crs=4326
    )
    already_ids = check_id(save_path)
    gdf = gdf[~gdf['uuid'].isin(already_ids)]

    ls_gadm = [gadm0, gadm1, gadm2, gadm3, gadm4, gadm5]
    input_dfs = []
    print('Splitting dataframe...')
    split_df(gdf, 20000, input_dfs)
    index = 0
    print('Start to spatial join...')
    for df in input_dfs:
        index += 1
        print('now:', index, 'total:', len(input_dfs))
        result = append_gadm(len(ls_gadm)-1, df, ls_gadm)
        if os.path.exists(save_path):
            temp = pd.read_csv(save_path)
            result = pd.concat([temp, result]).reset_index(drop=True)
        result.to_csv(save_path, index=False)