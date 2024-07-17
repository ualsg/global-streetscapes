import pandas as pd
import os
rootpath = 'manual_labels'

df = pd.read_csv(os.path.join(rootpath,'train/weather.csv'))

print(df['weather'].value_counts())

precipitation_states = ['foggy', 'rainy', 'snowy']
non_precipitation_states = ['clear', 'cloudy']

# Function to process dataset
def process_dataset(file_path, save_precipitation_path, save_non_precipitation_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Filter for precipitation and non-precipitation states
    df_precipitation = df[df['weather'].isin(precipitation_states)]
    df_precipitation['weather_precipitation'] = df_precipitation['weather']
    
    df_non_precipitation = df[df['weather'].isin(non_precipitation_states)]
    df_non_precipitation['weather_nonprecipitation'] = df_non_precipitation['weather']

    # Save to new files
    df_precipitation.to_csv(save_precipitation_path, index=False)
    df_non_precipitation.to_csv(save_non_precipitation_path, index=False)

# Process training set
process_dataset(os.path.join(rootpath,'train/weather.csv'),
                os.path.join(rootpath,'train/weather_precipitation.csv'),
                os.path.join(rootpath,'train/weather_nonprecipitation.csv')
               )

# Process test set
process_dataset(os.path.join(rootpath,'test/weather.csv'),
                os.path.join(rootpath,'test/weather_precipitation.csv'),
                os.path.join(rootpath,'test/weather_nonprecipitation.csv')
               )


# Test datasets to verify
precip = pd.read_csv(os.path.join(rootpath,'train/precipitation.csv'))
noprecip = pd.read_csv(os.path.join(rootpath,'train/non_precipitation.csv'))
print(f"TRAIN: {precip['weather'].value_counts()} \n \n {noprecip['weather'].value_counts()}")

# Test set
precip = pd.read_csv(os.path.join(rootpath,'test/precipitation.csv'))
noprecip = pd.read_csv(os.path.join(rootpath,'test/non_precipitation.csv'))
print(f"TRAIN: {precip['weather'].value_counts()} \n \n {noprecip['weather'].value_counts()}")
