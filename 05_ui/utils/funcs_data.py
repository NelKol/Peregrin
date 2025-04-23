import numpy as np
import pandas as pd
from math import floor, ceil



def dir_round(value, digits=3, direction='down'):
    if direction == 'up':
        return ceil(value * 10**digits) / 10**digits
    elif direction == 'down':
        return floor(value * 10**digits) / 10**digits
    elif direction == None:
        return round(value, digits)


def percentile_thresholding(df, column_name: str, values: tuple):
    try:
        lower_percentile, upper_percentile = values
        lower_threshold = np.percentile(df[column_name], lower_percentile)
        upper_threshold = np.percentile(df[column_name], upper_percentile)
        return df[(df[column_name] >= lower_threshold) & (df[column_name] <= upper_threshold)]
    except ValueError:
        return df

def literal_thresholding(df, column_name: str, values: tuple):
    lower_threshold, upper_threshold = values
    return df[(df[column_name] >= lower_threshold) & (df[column_name] <= upper_threshold)]

def dataframe_filter(df, df_filter):
    return df[df["TRACK_ID"].isin(df_filter["TRACK_ID"])]

def values_for_a_metric(df, metric):
    df.dropna()
    min_value = floor(df[metric].min())
    max_value = ceil(df[metric].max())
    return min_value, max_value


def try_convert_numeric(x):
    try:
        # Only process strings
        if isinstance(x, str):
            x_stripped = x.strip()
            num = float(x_stripped)
            if num.is_integer():
                return int(num)
            else:
                return num
        else:
            return x
    except ValueError:
        return x

def merge_dfs(dataframes, on):

    # Initialize the first DataFrame as the base for merging
    merged_df = dataframes[0].map(str)

    # Use a for loop to merge each subsequent DataFrame
    for df in dataframes[1:]:

        df = df.reset_index(drop=True)
        df = df.map(str)
        merge_columns = [col for col in df.columns if col not in merged_df.columns or col in on]
        merged_df = pd.merge(
            merged_df,
            df[merge_columns],  # Select only necessary columns from df
            on=on,
            how='outer'
        )
    
    merged_df = merged_df.map(try_convert_numeric)
    return merged_df

def butter(df):                                                                                      # Smoothing the raw dataframe

    float_columns = [ # Definition of unneccesary float columns in the df which are to be convertet to integers
    'ID', 
    'TRACK_ID', 
    'POSITION_T', 
    'FRAME'
    ]

    df = pd.DataFrame(df)  
    df = df.apply(pd.to_numeric, errors='coerce').dropna(subset=['POSITION_X', 'POSITION_Y', 'POSITION_T'])                                 # Gets rid of the multiple index rows by converting the values to a numeric type and then dropping the NaN values


    # For some reason, the y coordinates extracted from trackmate are mirrored. That ofcourse would not affect the statistical tests, only the data visualization. However, to not get mindfucked..
    # Reflect y-coordinates around the midpoint for the directionality to be accurate, according to the microscope videos.
    y_mid = (df['POSITION_Y'].min() + df['POSITION_Y'].max()) / 2
    df['POSITION_Y'] = 2 * y_mid - df['POSITION_Y']

    columns_list = df.columns.tolist()
    columns_list.remove('LABEL')

    df = df[columns_list]

    # Here we convert the unnecessary floats (from the list in which we defined them) to integers
    df[float_columns] = df[float_columns].astype(int)

    return df


def calculate_traveled_distances_for_each_cell_per_frame(df):

    if df.empty:
        return np.nan

    # Ensure the DataFrame is sorted properly by CONDITION, TRACK_ID, and POSITION_T (time)
    df_sorted = df.sort_values(by=['CONDITION', 'REPLICATE', 'POSITION_T'])

    # For each track (within each condition), shift the coordinates to get the "next" point
    next_POSITION_X = df_sorted.groupby(['CONDITION', 'REPLICATE', 'TRACK_ID'])['POSITION_X'].shift(-1)
    next_POSITION_Y = df_sorted.groupby(['CONDITION', 'REPLICATE', 'TRACK_ID'])['POSITION_Y'].shift(-1)
    # df_sorted['next_POSITION_Z'] = df_sorted.groupby(['CONDITION', 'TRACK_ID'])['POSITION_Z'].shift(-1)

    # Calculate the Euclidean distance for the XY plane between consecutive points
    df_sorted['DISTANCE'] = np.sqrt(
        (next_POSITION_X - df_sorted['POSITION_X'])**2 +
        (next_POSITION_Y - df_sorted['POSITION_Y'])**2
    )

    # Optionally, drop rows where the next value is missing (i.e. the last row per track)
    df_sorted['DISTANCE'] = df_sorted['DISTANCE'].fillna(0)
    df_result = df_sorted

    # Display the results
    return df_result

def calculate_direction_of_travel_for_each_cell_per_frame(df):
    directions = []
    for condition in df['CONDITION'].unique():
        unique_cond = df[df['CONDITION'] == condition]
        for replicate in unique_cond['REPLICATE'].unique():
            unique_rep = unique_cond[unique_cond['REPLICATE'] == replicate]
            for track_id in unique_rep['TRACK_ID'].unique():   
                unique_track = unique_rep[unique_rep['TRACK_ID'] == track_id]
                dx = unique_track['POSITION_X'].diff().iloc[1:]
                dy = unique_track['POSITION_Y'].diff().iloc[1:]
                rad = (np.arctan2(dy, dx))
                for i in range(len(rad)):
                    directions.append({
                        'CONDITION': unique_cond['CONDITION'].iloc[i],
                        'REPLICATE': unique_rep['REPLICATE'].iloc[i],
                        'TRACK_ID': unique_track['TRACK_ID'].iloc[i], 
                        'POSITION_T': unique_track['POSITION_T'].iloc[i], 
                        'DIRECTION_RAD': rad.iloc[i],
                        })
    directions_df = pd.DataFrame(directions)
    return directions_df

def calculate_track_length_net_distances_and_confinement_ratios_per_each_cell_per_frame(df):
    # Sort the DataFrame by CONDITION, REPLICATE, TRACK_ID, and POSITION_T
    df = df.sort_values(by=['CONDITION', 'REPLICATE', 'TRACK_ID', 'POSITION_T'])

    # Calculate cumulative track length for each track
    df['TRACK_LENGTH'] = df.groupby(['CONDITION', 'REPLICATE', 'TRACK_ID'])['DISTANCE'].cumsum()

    # Get the starting position for each track
    start_positions = df.groupby(['CONDITION', 'REPLICATE', 'TRACK_ID'])[['POSITION_X', 'POSITION_Y']].transform('first')

    # Calculate net distance from the starting position
    df['NET_DISTANCE'] = np.sqrt(
        (df['POSITION_X'] - start_positions['POSITION_X'])**2 +
        (df['POSITION_Y'] - start_positions['POSITION_Y'])**2
    )

    # Calculate confinement ratio
    df['CONFINEMENT_RATIO'] = df['NET_DISTANCE'] / df['TRACK_LENGTH']
    df['CONFINEMENT_RATIO'] = df['CONFINEMENT_RATIO'].fillna(0)  # Handle division by zero

    return df

def calculate_track_lengths_and_net_distances(df):
    if df.empty:
        return np.nan

    # Ensure the data is sorted
    df_sorted = df.sort_values(by=['CONDITION', 'REPLICATE', 'TRACK_ID', 'FRAME'])
    
    # Sum the DISTANCE values for each track within each condition
    track_length_df = df_sorted.groupby(['CONDITION', 'REPLICATE', 'TRACK_ID'], as_index=False)['DISTANCE'].sum().rename(columns={'DISTANCE': 'TRACK_LENGTH'})

    # Calculate the net distance for each track
    def net_distance_per_track(track_df):
        start_position = track_df.iloc[0][['POSITION_X', 'POSITION_Y']].values
        end_position = track_df.iloc[-1][['POSITION_X', 'POSITION_Y']].values
        return np.sqrt((end_position[0] - start_position[0])**2 + (end_position[1] - start_position[1])**2)
    
    net_distances = df_sorted.groupby(['CONDITION', 'REPLICATE', 'TRACK_ID']).apply(net_distance_per_track).reset_index(name='NET_DISTANCE')

    # Merge the track lengths and net distances
    track_lengths_and_net_distances_df = pd.merge(track_length_df, net_distances, on=['CONDITION', 'REPLICATE', 'TRACK_ID'], how='outer')

    return track_lengths_and_net_distances_df

def calculate_confinement_ratio_for_each_cell(df):
    # Calculate the confinement ratio
    df['CONFINEMENT_RATIO'] = df['NET_DISTANCE'] / df['TRACK_LENGTH']
    Track_stats_df = df[['CONDITION', 'REPLICATE', 'TRACK_ID', 'CONFINEMENT_RATIO']]
    return pd.DataFrame(Track_stats_df)

def calculate_distances_per_frame(df):
    # df['POSITION_T'] = df['POSITION_T'].astype(int)  # Convert POSITION_T to integers
    min_distance_per_frame = df.groupby(['CONDITION', 'REPLICATE', 'POSITION_T'])['DISTANCE'].min().reset_index()
    min_distance_per_frame.rename(columns={'DISTANCE': 'min_DISTANCE'}, inplace=True)
    max_distance_per_frame = df.groupby(['CONDITION', 'REPLICATE', 'POSITION_T'])['DISTANCE'].max().reset_index()
    max_distance_per_frame.rename(columns={'DISTANCE': 'max_DISTANCE'}, inplace=True)
    mean_distances_per_frame = df.groupby(['CONDITION', 'REPLICATE', 'POSITION_T'])['DISTANCE'].mean().reset_index()
    mean_distances_per_frame.rename(columns={'DISTANCE': 'MEAN_DISTANCE'}, inplace=True)
    std_deviation_distances_per_frame = df.groupby(['CONDITION', 'REPLICATE', 'POSITION_T'])['DISTANCE'].std().reset_index()
    std_deviation_distances_per_frame.rename(columns={'DISTANCE': 'STD_DEVIATION_distances'}, inplace=True)
    median_distances_per_frame = df.groupby(['CONDITION', 'REPLICATE', 'POSITION_T'])['DISTANCE'].median().reset_index()
    median_distances_per_frame.rename(columns={'DISTANCE': 'MEDIAN_DISTANCE'}, inplace=True)

    merging = [min_distance_per_frame, max_distance_per_frame, mean_distances_per_frame, std_deviation_distances_per_frame, median_distances_per_frame]
    merged = merge_dfs(merging, on=['CONDITION', 'REPLICATE', 'POSITION_T'])
    merged = merged.sort_values(by=['CONDITION', 'REPLICATE', 'POSITION_T'])
    return merged

def weighted_mean_direction(angles, weights):
    weighted_sin = np.average(np.sin(angles), weights=weights)
    weighted_cos = np.average(np.cos(angles), weights=weights)
    return np.arctan2(weighted_sin, weighted_cos)

def weighted_std_deviation(angles, weights, mean_direction_rad):
    weighted_sin = np.average(np.sin(angles - mean_direction_rad), weights=weights)
    weighted_cos = np.average(np.cos(angles - mean_direction_rad), weights=weights)
    return np.sqrt(weighted_sin**2 + weighted_cos**2)

def weighted_median_direction(angles, weights):
    sorted_angles = np.sort(angles)
    sorted_weights = np.array(weights)[np.argsort(angles)]
    cumsum_weights = np.cumsum(sorted_weights)
    midpoint = np.sum(weights) / 2
    idx = np.searchsorted(cumsum_weights, midpoint)
    return sorted_angles[idx]

def calculate_absolute_directions_per_cell(df):
    mean_direction_rad = df.groupby(['CONDITION', 'REPLICATE', 'TRACK_ID'])['DIRECTION_RAD'].apply(lambda angles: np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))
    mean_direction_deg = np.degrees(mean_direction_rad) % 360
    std_deviation_rad = df.groupby(['CONDITION', 'REPLICATE', 'TRACK_ID'])['DIRECTION_RAD'].apply(lambda angles: np.sqrt(np.mean(np.cos(angles))**2 + np.mean(np.sin(angles))**2))
    std_deviation_deg = np.degrees(std_deviation_rad) % 360
    median_direction_rad = df.groupby(['CONDITION', 'REPLICATE', 'TRACK_ID'])['DIRECTION_RAD'].apply(lambda angles: np.arctan2(np.median(np.sin(angles)), np.median(np.cos(angles))))
    median_direction_deg = np.degrees(median_direction_rad) % 360

    return pd.DataFrame({
        'CONDITION': mean_direction_rad.index.get_level_values('CONDITION'),
        'REPLICATE': mean_direction_rad.index.get_level_values('REPLICATE'),
        'TRACK_ID': mean_direction_rad.index.get_level_values('TRACK_ID'),
        'MEAN_DIRECTION_DEG': mean_direction_deg,
        'STD_DEVIATION_DEG': std_deviation_deg,
        'MEDIAN_DIRECTION_DEG': median_direction_deg,
        'MEAN_DIRECTION_RAD': mean_direction_rad,
        'STD_DEVIATION_RAD': std_deviation_rad,
        'MEDIAN_DIRECTION_RAD': median_direction_rad
    }).reset_index(drop=True)

def calculate_mean_median_std_cr_nd_tl_per_frame(df):
    # Define the grouping columns
    group_cols = ['CONDITION', 'REPLICATE', 'POSITION_T']

    # Calculate mean values per frame
    mean_values = df.groupby(group_cols).agg({
        'TRACK_LENGTH': 'mean',
        'NET_DISTANCE': 'mean',
        'CONFINEMENT_RATIO': 'mean'
    }).rename(columns={
        'TRACK_LENGTH': 'MEAN_TRACK_LENGTH',
        'NET_DISTANCE': 'MEAN_NET_DISTANCE',
        'CONFINEMENT_RATIO': 'MEAN_CONFINEMENT_RATIO'
    }).reset_index()

    # Calculate median values per frame
    median_values = df.groupby(group_cols).agg({
        'TRACK_LENGTH': 'median',
        'NET_DISTANCE': 'median',
        'CONFINEMENT_RATIO': 'median'
    }).rename(columns={
        'TRACK_LENGTH': 'MEDIAN_TRACK_LENGTH',
        'NET_DISTANCE': 'MEDIAN_NET_DISTANCE',
        'CONFINEMENT_RATIO': 'MEDIAN_CONFINEMENT_RATIO'
    }).reset_index()

    # Calculate standard deviation values per frame
    std_values = df.groupby(group_cols).agg({
        'TRACK_LENGTH': 'std',
        'NET_DISTANCE': 'std',
        'CONFINEMENT_RATIO': 'std'
    }).rename(columns={
        'TRACK_LENGTH': 'STD_TRACK_LENGTH',
        'NET_DISTANCE': 'STD_NET_DISTANCE',
        'CONFINEMENT_RATIO': 'STD_CONFINEMENT_RATIO'
    }).reset_index()

    min_values = df.groupby(group_cols).agg({
        'TRACK_LENGTH': 'min',
        'NET_DISTANCE': 'min',
        'CONFINEMENT_RATIO': 'min'
    }).rename(columns={
        'TRACK_LENGTH': 'MIN_TRACK_LENGTH',
        'NET_DISTANCE': 'MIN_NET_DISTANCE',
        'CONFINEMENT_RATIO': 'MIN_CONFINEMENT_RATIO'
    }).reset_index()

    max_values = df.groupby(group_cols).agg({
        'TRACK_LENGTH': 'max',
        'NET_DISTANCE': 'max',
        'CONFINEMENT_RATIO': 'max'
    }).rename(columns={
        'TRACK_LENGTH': 'MAX_TRACK_LENGTH',
        'NET_DISTANCE': 'MAX_NET_DISTANCE',
        'CONFINEMENT_RATIO': 'MAX_CONFINEMENT_RATIO'
    }).reset_index()

    # Merge the mean, median, and std DataFrames on the grouping columns
    stats_df = merge_dfs([mean_values, median_values, std_values, min_values, max_values], on=group_cols)

    return stats_df

def calculate_absolute_directions_per_frame(df):
    grouped = df.groupby(['CONDITION', 'REPLICATE', 'POSITION_T'])
    mean_direction_rad = grouped['DIRECTION_RAD'].apply(lambda angles: np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))
    mean_direction_deg = np.degrees(mean_direction_rad) % 360
    std_deviation_rad = grouped['DIRECTION_RAD'].apply(lambda angles: np.sqrt(np.mean(np.cos(angles))**2 + np.mean(np.sin(angles))**2))
    std_deviation_deg = np.degrees(std_deviation_rad) % 360
    median_direction_rad = grouped['DIRECTION_RAD'].apply(lambda angles: np.arctan2(np.median(np.sin(angles)), np.median(np.cos(angles))))
    median_direction_deg = np.degrees(median_direction_rad) % 360
    
    result_df = pd.DataFrame({
        'CONDITION': mean_direction_rad.index.get_level_values('CONDITION'),
        'REPLICATE': mean_direction_rad.index.get_level_values('REPLICATE'),
        'POSITION_T': mean_direction_rad.index.get_level_values('POSITION_T'),
        'MEAN_DIRECTION_DEG': mean_direction_deg, 
        'STD_DEVIATION_DEG': std_deviation_deg, 
        'MEDIAN_DIRECTION_DEG': median_direction_deg, 
        'MEAN_DIRECTION_RAD': mean_direction_rad, 
        'STD_DEVIATION_RAD': std_deviation_rad, 
        'MEDIAN_DIRECTION_RAD': median_direction_rad
    }).reset_index(drop=True)
    
    return result_df

def calculate_number_of_frames_per_cell(spot_stats_df):
    # Count the number of frames for each TRACK_ID in the spot_stats_df
    frames_per_track = spot_stats_df.groupby(['CONDITION', 'REPLICATE', 'TRACK_ID']).size().reset_index(name='TRACK_POINTS')
    return frames_per_track

def calculate_speed(df, variable):

    if isinstance(variable, list):
        variable1, variable2 = variable
        min_speed_microns_min = df.groupby(['CONDITION', variable1, variable2])['DISTANCE'].min().reset_index()
        min_speed_microns_min.rename(columns={'DISTANCE': 'SPEED_MIN'}, inplace=True)
        max_speed_microns_min = df.groupby(['CONDITION', variable1, variable2])['DISTANCE'].max().reset_index()
        max_speed_microns_min.rename(columns={'DISTANCE': 'SPEED_MAX'}, inplace=True)
        mean_speed_microns_min = df.groupby(['CONDITION', variable1, variable2])['DISTANCE'].mean().reset_index()
        mean_speed_microns_min.rename(columns={'DISTANCE': 'SPEED_MEAN'}, inplace=True)
        std_deviation_speed_microns_min = df.groupby(['CONDITION', variable1, variable2])['DISTANCE'].std().reset_index()
        std_deviation_speed_microns_min.rename(columns={'DISTANCE': 'SPEED_STD_DEVIATION'}, inplace=True)
        median_speed_microns_min = df.groupby(['CONDITION', variable1, variable2])['DISTANCE'].median().reset_index()
        median_speed_microns_min.rename(columns={'DISTANCE': 'SPEED_MEDIAN'}, inplace=True)

        merging = [min_speed_microns_min, max_speed_microns_min, mean_speed_microns_min, std_deviation_speed_microns_min, median_speed_microns_min]
        merged = merge_dfs(merging, on=['CONDITION', variable1, variable2])
        
    else:
        min_speed_microns_min = df.groupby(['CONDITION', variable])['DISTANCE'].min().reset_index()
        min_speed_microns_min.rename(columns={'DISTANCE': 'SPEED_MIN'}, inplace=True)
        max_speed_microns_min = df.groupby(['CONDITION', variable])['DISTANCE'].max().reset_index()
        max_speed_microns_min.rename(columns={'DISTANCE': 'SPEED_MAX'}, inplace=True)
        mean_speed_microns_min = df.groupby(['CONDITION', variable])['DISTANCE'].mean().reset_index()
        mean_speed_microns_min.rename(columns={'DISTANCE': 'SPEED_MEAN'}, inplace=True)
        std_deviation_speed_microns_min = df.groupby(['CONDITION', variable])['DISTANCE'].std().reset_index()
        std_deviation_speed_microns_min.rename(columns={'DISTANCE': 'SPEED_STD_DEVIATION'}, inplace=True)
        median_speed_microns_min = df.groupby(['CONDITION', variable])['DISTANCE'].median().reset_index()
        median_speed_microns_min.rename(columns={'DISTANCE': 'SPEED_MEDIAN'}, inplace=True)
        
        merging = [min_speed_microns_min, max_speed_microns_min, mean_speed_microns_min, std_deviation_speed_microns_min, median_speed_microns_min]
        merged = merge_dfs(merging, on=['CONDITION', variable])

    return pd.DataFrame(merged)

def get_cond_repl(df):
    # Get unique conditions from the DataFrame
    dictionary = {'all': ['all']}
    

    conditions = df['CONDITION'].unique()
    for condition in conditions:
        # Get unique replicates for each condition
        replicates_list = ['all']
        replicates = df[df['CONDITION'] == condition]['REPLICATE'].unique()
        for replicate in replicates.tolist():
            replicate = str(replicate)
            replicates_list.append(replicate)
            
        dictionary.update({str(condition): replicates_list})
    return dictionary