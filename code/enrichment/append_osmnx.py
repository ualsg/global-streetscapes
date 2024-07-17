import osmnx as ox
import pandas as pd
import geopandas as gp
import numpy as np
import os
from pathlib import Path

def snap_photos_to_roads(points, roads, tolerance):
    """
    inputs:
        points = GeoDataFrame of points to be snapped
        roads = GeoDataFrame of lines where points are to be snapped to
        tolerance = only if a point is within this distance (meters) of a line, it will be snapped to the line
    output:
        a GeoDataFrame of snapped points
    """
    
    points_bbox = points.bounds + [-tolerance, -tolerance, tolerance, tolerance] # construct a bounding box around each point according to the set tolerance
    
    hits = points_bbox.apply(lambda row: list(roads.sindex.intersection(row)), axis=1) # get a list of the lines that intersect with each bbox
    
    # a df that expands the hits df so there is one row for every (point index, intersecting line index) pair
    # implicitly drops those bboxes that didn’t have any lines within their window
    tmp = pd.DataFrame({
        # index of point
        "pt_idx": np.repeat(hits.index, hits.apply(len)),
        # ordinal position of line - access via iloc later
        "line_i": np.concatenate(hits.values)
    })
    
    # if there isn't road nearby any point at all:
    if tmp.empty:
        empty_df = gp.GeoDataFrame(columns=['geometry'], geometry='geometry')
        return empty_df
    
    else: 
        # join tmp with roads on line_i, use reset_index() to give us the ordinal position of each line
        # rename width as road_width
        roads_2 = roads.rename(columns={"width": "road_width"})
        tmp = tmp.join(roads_2.reset_index(drop=True), on="line_i")
        
        # join with the original points, rename the point geometry as "point"
        points_2 = points.rename(columns={"geometry": "og_point"})
        tmp = tmp.join(points_2, on="pt_idx")
        
        # convert back to a GeoDataFrame, so we can do spatial ops
        tmp = gp.GeoDataFrame(tmp, geometry="geometry", crs=points.crs)
        
        # calculate the distance between each point and its associated lines
        tmp["snap_dist"] = tmp.geometry.distance(gp.GeoSeries(tmp['og_point']))
        
        # discard any lines whose distnaces from points are > tolerance
        tmp = tmp.loc[tmp.snap_dist <= tolerance]
        
        # sort on ascending snap distance, so that closest goes to top
        tmp = tmp.sort_values(by=["snap_dist"])
        
        # group by the index of the points and take the first, which is the closest line 
        closest = tmp.groupby("pt_idx").first()
        
        # construct a GeoDataFrame of the closest lines
        closest = gp.GeoDataFrame(closest, geometry="geometry", crs=points.crs)
        
        # position of nearest point from start of the line
        pos = closest.geometry.project(gp.GeoSeries(closest['og_point']))
        
        # get new point geometry
        snapped_pts = closest.geometry.interpolate(pos)
        
        # rename closest's geometry column
        closest_2 = closest.rename(columns={"geometry": "line_geometry"})

        # create a new GeoDataFrame from closest_2 and the new point geometries (which will be called "geometry")
        snapped = gp.GeoDataFrame(closest_2,geometry=snapped_pts).reset_index()
        
        return snapped
    

def append_road_info(city, images, snap_tolerance, save_folder, save_folder_osm):
    points = images[images['city_id'] == city]
    location = (points.iloc[0]['city_lat'], points.iloc[0]['city_lon'])
    print(location)
    G = ox.graph_from_point(location, dist=3000, network_type='all')
    G = ox.get_undirected(G)
    df_G = ox.graph_to_gdfs(G)[1]
    roads = df_G.reset_index().reset_index()
    roads_proj = ox.project_gdf(df_G).reset_index().reset_index()
    roads['geometry_wkt'] = roads['geometry'].to_wkt()
    roads_proj['geometry_wkt'] = roads_proj['geometry'].to_wkt()
    iso3 = points.iloc[0]['iso3']
    cityname = points.iloc[0]['city_ascii']
    save_path1 = os.path.join(save_folder_osm, f'{city}_osm.csv')
    save_path2 = os.path.join(save_folder_osm, f'{city}_osm_proj.csv')
    roads.drop(columns=['geometry']).to_csv(save_path1)
    roads_proj.drop(columns=['geometry']).to_csv(save_path2)
    print(f'Downloaded successfully OSM roads for {iso3}_{cityname}', city)

    points_proj = ox.project_gdf(points)
    snapped = snap_photos_to_roads(points_proj, roads_proj, snap_tolerance)
    if snapped.empty:
        print(f'No road near any images for {iso3}_{cityname}', city)
    else:
        snapped['l_geom_wkt'] = snapped['line_geometry'].to_wkt()
        snapped['og_pt_wkt'] = snapped['og_point'].to_wkt()
        snapped['snp_pt_wkt'] = snapped['geometry'].to_wkt()
        save_path = os.path.join(save_folder, f'{city}_snapped.csv')
        pd.DataFrame(snapped.drop(columns=['geometry', 'line_geometry', 'og_point'])).to_csv(save_path)
        print(f'Appended successfully for {iso3}_{cityname}', city)
    
def check_id(save_folder):
    ids = set()
    for name in os.listdir(save_folder):
        if name != '.DS_Store':
            ids.add(name.split('_')[0])
    return ids


if __name__ == '__main__':

    df = pd.read_csv('./sample_data/01_simplemaps.csv')
    meta = pd.read_csv('./sample_data/02_metadata_common_attributes.csv')[['uuid', 'lat', 'lon']]
    df = df.merge(meta, on='uuid', how='left')
    save_folder = './sample_data/snapped'
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    save_folder_osm ='./sample_data/osm'
    Path(save_folder_osm).mkdir(parents=True, exist_ok=True)

    images = gp.GeoDataFrame(
        df, geometry=gp.points_from_xy(df.lon, df.lat), crs=4326
    )

    snap_tolerance = 10
    already_id = check_id(save_folder)
    total = images['city_id'].nunique()
    cities = images['city_id'].unique().tolist()

    index = 0

    for city in cities:
        
        if str(city) in already_id:
            continue

        index += 1
        append_road_info(city, images, snap_tolerance, save_folder, save_folder_osm)
        print('Now:', index, (total-len(already_id)), 'already:', index + len(already_id))
    
    size = len(check_id(save_folder))
    print('Number of cities with OSM data:', size, '/', total)
    print('Done')