import numpy as np
import pandas as pd


def calculate_traveled_distances_for_each_cell_per_frame(df):

    # Calculate distances between consecutive frames
    if df.empty:
            return np.nan
    
    def distance_per_frame(group):
        group = group.copy()
        # Compute the Euclidean distance between consecutive frames
        group['DISTANCE'] = np.sqrt(
            (group['X coordinate'].diff() ** 2) +
            (group['POSITION_Y'].diff() ** 2)
        )
        return group
    
    # Ensure TRACK_ID is not used as index in grouping; Calculate distances; Drop rows where DISTANCE is NaN (first frame of each TRACK_ID group)
    result_df = df.reset_index(drop=True).groupby('TRACK_ID', as_index=False).apply(distance_per_frame).dropna(subset='DISTANCE')

    return pd.DataFrame(result_df)

def calculate_track_lengths_and_net_distances(df):

    # Convert 'Track ID' to numeric (if it's not already)
    df['TRACK_ID'] = pd.to_numeric(df['TRACK_ID'], errors='coerce')
    
    # Making sure that no empty lines are created in the DataFrame
    if df.empty:
        return np.nan
    
    # Sum distances per Track ID
    track_lengths = df.groupby('TRACK_ID')['DISTANCE'].sum().reset_index()

    # Rename columns for clarity
    track_lengths.columns = ['TRACK_ID', 'Track length']
    
    # Get the start and end positions for each track, calculate the enclosed distance and group by 'TRACK_ID'
    def calculate_distance(start, end):
        return np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
    
    def net_distance_per_track(track_df):
        if track_df.empty:
            return np.nan
        start_position = track_df.iloc[0][['X coordinate', 'POSITION_Y']].values
        end_position = track_df.iloc[-1][['X coordinate', 'POSITION_Y']].values
        return calculate_distance(start_position, end_position)

    net_distances = df.groupby('TRACK_ID').apply(net_distance_per_track).reset_index(name='Net distance')

    track_lengths_and_net_distances = pd.merge(track_lengths, net_distances, on='TRACK_ID', how='outer')

    # Return the results
    return pd.DataFrame(track_lengths_and_net_distances)

def calculate_confinement_ratio_for_each_cell(df):
    # Calculate the confinement ratio
    df['Confinement ratio'] = df['Net distance'] / df['Track length']
    Track_stats_df = df[['TRACK_ID','Confinement ratio']]
    return pd.DataFrame(Track_stats_df)

def calculate_distances_per_frame(df):
    min_distance_per_frame = df.groupby('POSITION_T')['DISTANCE'].min().reset_index()
    min_distance_per_frame.rename(columns={'DISTANCE': 'Distance min'}, inplace=True)
    max_distance_per_frame = df.groupby('POSITION_T')['DISTANCE'].max().reset_index()
    max_distance_per_frame.rename(columns={'DISTANCE': 'max_DISTANCE'}, inplace=True)
    mean_distances_per_frame = df.groupby('POSITION_T')['DISTANCE'].mean().reset_index()
    mean_distances_per_frame.rename(columns={'DISTANCE': 'MEAN_DISTANCE'}, inplace=True)
    std_deviation_distances_per_frame = df.groupby('POSITION_T')['DISTANCE'].std().reset_index()
    std_deviation_distances_per_frame.rename(columns={'DISTANCE': 'STD_DEVIATION_distances'}, inplace=True)
    median_distances_per_frame = df.groupby('POSITION_T')['DISTANCE'].median().reset_index()
    median_distances_per_frame.rename(columns={'DISTANCE': 'MEDIAN_DISTANCE'}, inplace=True)
    merge = pd.merge(min_distance_per_frame, max_distance_per_frame, on='POSITION_T')
    merge = pd.merge(merge, mean_distances_per_frame, on='POSITION_T')
    merge = pd.merge(merge, std_deviation_distances_per_frame, on='POSITION_T')
    merged = pd.merge(merge, median_distances_per_frame, on='POSITION_T')
    return pd.DataFrame(merged)

def calculate_direction_of_travel_for_each_cell_per_frame(df):

    directions = []
    for track_id in df['TRACK_ID'].unique():
        track_data = df[df['TRACK_ID'] == track_id]
        dx = track_data['X coordinate'].diff().iloc[1:]
        dy = track_data['POSITION_Y'].diff().iloc[1:]
        rad = (np.arctan2(dy, dx))
        for i in range(len(rad)):
            directions.append({
                'TRACK_ID': track_id, 
                'POSITION_T': track_data['POSITION_T'].iloc[i + 1], 
                'DIRECTION_RAD': rad.iloc[i]
                })
    return pd.DataFrame(directions)

def weighted_mean_direction(angles, weights):
    weighted_sin = np.average(np.sin(angles), weights=weights)
    weighted_cos = np.average(np.cos(angles), weights=weights)
    return np.arctan2(weighted_sin, weighted_cos)
    
def weighted_std_deviation(angles, weights, mean_direction_rad):
    weighted_sin = np.average(np.sin(angles - mean_direction_rad), weights=weights)
    weighted_cos = np.average(np.cos(angles - mean_direction_rad), weights=weights)
    return np.sqrt(weighted_sin**2 + weighted_cos**2)
    
def weighted_median_direction(angles, weights):
    # Weighted median approximation: using percentile function from scipy
    from scipy.stats import cumfreq
    sorted_angles = np.sort(angles)
    sorted_weights = np.array(weights)[np.argsort(angles)]
    cumsum_weights = np.cumsum(sorted_weights)
    midpoint = np.sum(weights) / 2
    idx = np.searchsorted(cumsum_weights, midpoint)
    return sorted_angles[idx]

def calculate_absolute_directions_per_cell(df):
    mean_direction_rad = df.groupby('TRACK_ID')['DIRECTION_RAD'].apply(lambda angles: np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))
    mean_direction_deg = np.degrees(mean_direction_rad) % 360
    std_deviation_rad = df.groupby('TRACK_ID')['DIRECTION_RAD'].apply(lambda angles: np.sqrt(np.mean(np.cos(angles))**2 + np.mean(np.sin(angles))**2))
    std_deviatin_deg = np.degrees(std_deviation_rad) % 360
    median_direction_rad = df.groupby('TRACK_ID')['DIRECTION_RAD'].apply(lambda angles: np.arctan2(np.median(np.sin(angles)), np.median(np.cos(angles))))
    median_direction_deg = np.degrees(median_direction_rad) % 360

    return pd.DataFrame({
        'TRACK_ID': mean_direction_rad.index, 
        'Direction mean (deg)': mean_direction_deg, 
        'Direction std (deg)': std_deviatin_deg, 
        'MEDIAN_DIRECTION_DEG': median_direction_deg, 
        'Direction mean (rad)': mean_direction_rad, 
        'Direction std (rad)': std_deviation_rad, 
        'MEADIAN_DIRECTION_RAD': median_direction_rad
        }).reset_index(drop=True)

def calculate_weighted_directions_per_cell(df):
    df = df.dropna(subset=['TRACK_ID'])

    confinement_ratio_weighted_mean_direction_rad = df['Direction mean (rad)'] * df['Confinement ratio'] / df['Confinement ratio']
    confinement_ratio_weighted_mean_direction_deg = np.degrees(confinement_ratio_weighted_mean_direction_rad) % 360
    confinement_ratio_weighted_std_deviation_rad = df['Direction std (rad)'] * df['Confinement ratio'] / df['Confinement ratio']	
    confinement_ratio_weighted_std_deviation_deg = np.degrees(confinement_ratio_weighted_std_deviation_rad) % 360
    confinement_ratio_weighted_median_direction_rad = df['MEADIAN_DIRECTION_RAD'] * df['Confinement ratio'] / df['Confinement ratio']
    confinement_ratio_weighted_median_direction_deg = np.degrees(confinement_ratio_weighted_median_direction_rad) % 360

    net_distance_weighted_mean_direction_rad = df['Direction mean (rad)'] * df['Net distance'] / df['Net distance']
    net_distance_weighted_mean_direction_deg = np.degrees(net_distance_weighted_mean_direction_rad) % 360
    net_distance_weighted_std_deviation_rad = df['Direction std (rad)'] * df['Net distance'] / df['Net distance']
    net_distance_weighted_std_deviation_deg = np.degrees(net_distance_weighted_std_deviation_rad) % 360
    net_distance_weighted_median_direction_rad = df['MEADIAN_DIRECTION_RAD'] * df['Net distance'] / df['Net distance']
    net_distance_weighted_median_direction_deg = np.degrees(net_distance_weighted_median_direction_rad) % 360    

    return pd.DataFrame({
    'TRACK_ID': df.index, 
    'Direction mean (deg)_weight_confinement': confinement_ratio_weighted_mean_direction_deg, 
    'Direction std (deg)_weight_confinement': confinement_ratio_weighted_std_deviation_deg, 
    'MEDIAN_DIRECTION_DEG_weight_confinement': confinement_ratio_weighted_median_direction_deg, 
    'Direction mean (rad)_weight_confinement': confinement_ratio_weighted_mean_direction_rad, 
    'Direction std (rad)_weight_confinement': confinement_ratio_weighted_std_deviation_rad, 
    'MEADIAN_DIRECTION_RAD_weight_confinement': confinement_ratio_weighted_median_direction_rad,
    'Direction mean (deg)_weight_net_dis': net_distance_weighted_mean_direction_deg, 
    'Direction std (deg)_weight_net_dis': net_distance_weighted_std_deviation_deg, 
    'MEDIAN_DIRECTION_DEG_weight_net_dis': net_distance_weighted_median_direction_deg, 
    'Direction mean (rad)_weight_net_dis': net_distance_weighted_mean_direction_rad, 
    'Direction std (rad)_weight_net_dis': net_distance_weighted_std_deviation_rad, 
    'MEADIAN_DIRECTION_RAD_weight_net_dis': net_distance_weighted_median_direction_rad
    }).reset_index(drop=True)

def calculate_absolute_directions_per_frame(df):
    grouped = df.groupby('POSITION_T')
    mean_direction_rad = grouped['DIRECTION_RAD'].apply(lambda angles: np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))
    mean_direction_deg = np.degrees(mean_direction_rad) % 360
    std_deviation_rad = grouped['DIRECTION_RAD'].apply(lambda angles: np.sqrt(np.mean(np.cos(angles))**2 + np.mean(np.sin(angles))**2))
    std_deviatin_deg = np.degrees(std_deviation_rad) % 360
    median_direction_rad = grouped['DIRECTION_RAD'].apply(lambda angles: np.arctan2(np.median(np.sin(angles)), np.median(np.cos(angles))))
    median_direction_deg = np.degrees(median_direction_rad) % 360
    return pd.DataFrame({
        'POSITION_T': mean_direction_rad.index, 
        'Direction mean (deg)': mean_direction_deg, 
        'Direction std (deg)': std_deviatin_deg, 
        'MEDIAN_DIRECTION_DEG': median_direction_deg, 
        'Direction mean (rad)': mean_direction_rad, 
        'Direction std (rad)': std_deviation_rad, 
        'MEADIAN_DIRECTION_RAD': median_direction_rad
        }).reset_index(drop=True)

def calculate_weighted_directions_per_frame(df):
    grouped = df.groupby('POSITION_T')
    
    # Compute weighted metrics
    weighted_mean_direction_rad = grouped.apply(lambda x: weighted_mean_direction(x['DIRECTION_RAD'], x['DISTANCE'])).reset_index(level=0, drop=True)
    weighted_mean_direction_deg = np.degrees(weighted_mean_direction_rad) % 360
    
    weighted_std_deviation_rad = grouped.apply(lambda x: weighted_std_deviation(x['DIRECTION_RAD'], x['DISTANCE'], weighted_mean_direction(x['DIRECTION_RAD'], x['DISTANCE']))).reset_index(level=0, drop=True)
    weighted_std_deviation_deg = np.degrees(weighted_std_deviation_rad) % 360
    
    weighted_median_direction_rad = grouped.apply(lambda x: weighted_median_direction(x['DIRECTION_RAD'], x['DISTANCE'])).reset_index(level=0, drop=True)
    weighted_median_direction_deg = np.degrees(weighted_median_direction_rad) % 360
    
    result_df = pd.DataFrame({
        'POSITION_T': weighted_mean_direction_rad.index,
        'Direction mean (deg)_weight_mean_dis': weighted_mean_direction_deg,
        'Direction std (deg)_weight_mean_dis': weighted_std_deviation_deg,
        'MEDIAN_DIRECTION_DEG_weight_mean_dis': weighted_median_direction_deg,
        'Direction mean (rad)_weight_mean_dis': weighted_mean_direction_rad,
        'Direction std (rad)_weight_mean_dis': weighted_std_deviation_rad,
        'Direction median (rad)_weight_mean_dis': weighted_median_direction_rad
    }).reset_index(drop=True)

    result_df['POSITION_T'] = result_df['POSITION_T'] + 1
    
    return result_df

def calculate_number_of_frames_per_cell(spot_stats_df):
    # Count the number of frames for each TRACK_ID in the Spot_stats2_df
    frames_per_track = (spot_stats_df.groupby("TRACK_ID").size().reset_index(name="NUM_FRAMES"))

    return frames_per_track

def calculate_speed(df, variable):

    min_speed_microns_min = df.groupby(variable)['DISTANCE'].min().reset_index()
    min_speed_microns_min.rename(columns={'DISTANCE': 'Speed min'}, inplace=True)
    max_speed_microns_min = df.groupby(variable)['DISTANCE'].max().reset_index()
    max_speed_microns_min.rename(columns={'DISTANCE': 'Speed max'}, inplace=True)
    mean_speed_microns_min = df.groupby(variable)['DISTANCE'].mean().reset_index()
    mean_speed_microns_min.rename(columns={'DISTANCE': 'Speed mean'}, inplace=True)
    std_deviation_speed_microns_min = df.groupby(variable)['DISTANCE'].std().reset_index()
    std_deviation_speed_microns_min.rename(columns={'DISTANCE': 'Speed std'}, inplace=True)
    median_speed_microns_min = df.groupby(variable)['DISTANCE'].median().reset_index()
    median_speed_microns_min.rename(columns={'DISTANCE': 'Speed median'}, inplace=True)
    merge = pd.merge(min_speed_microns_min, max_speed_microns_min, on=variable)
    merge = pd.merge(merge, mean_speed_microns_min, on=variable)
    merge = pd.merge(merge, std_deviation_speed_microns_min, on=variable)
    merged = pd.merge(merge, median_speed_microns_min, on=variable)

    return pd.DataFrame(merged)


def merge_dfs(dataframes, on):

    # Initialize the first DataFrame as the base for merging
    merged_df = dataframes[0]

    # Use a for loop to merge each subsequent DataFrame
    for df in dataframes[1:]:
        # Merge, avoiding duplicate columns
        merge_columns = [col for col in df.columns if col not in merged_df.columns or col in on]
        merged_df = pd.merge(
            merged_df,
            df[merge_columns],  # Select only necessary columns from df
            on=on,
            how='outer'
        )

    return merged_df


df = pd.read_csv('buttered.csv') # Load the Spot_stats.csv file into a DataFrame


distances_for_each_cell_per_frame_df = calculate_traveled_distances_for_each_cell_per_frame(df) # Call the function to calculate distances for each cell per frame and create the Spot_statistics .csv file
direction_for_each_cell_per_frame_df = calculate_direction_of_travel_for_each_cell_per_frame(df) # Call the function to calculate direction_for_each_cell_per_frame_df

Spot_stats_dfs = [df, distances_for_each_cell_per_frame_df, direction_for_each_cell_per_frame_df]

Spot_stats = merge_dfs(Spot_stats_dfs, on=['TRACK_ID', 'POSITION_T'])
Spot_stats.to_csv('Spot_stats.csv') # Saving the Spot_stats DataFrame into a newly created .csv file


tracks_lengths_and_net_distances_df = calculate_track_lengths_and_net_distances(Spot_stats) # Calling function to calculate the total distance traveled for each cell from the distances_for_each_cell_per_frame_df
confinement_ratios_df = calculate_confinement_ratio_for_each_cell(tracks_lengths_and_net_distances_df) # Call the function to calculate confinement ratios from the Track_statistics1_df and write it into the Track_statistics1_df
track_directions_df = calculate_absolute_directions_per_cell(Spot_stats) # Call the function to calculate directions_per_cell_df
frames_per_track = calculate_number_of_frames_per_cell(Spot_stats)
speeds_per_cell = calculate_speed(Spot_stats, 'TRACK_ID')

Track_stats_dfs = [tracks_lengths_and_net_distances_df, confinement_ratios_df, track_directions_df, frames_per_track, speeds_per_cell]
Track_stats = merge_dfs(Track_stats_dfs, on='TRACK_ID')
Track_stats.to_csv('Track_stats.csv', index=False) # Save the Track_stats created DataFrame into a newly created Track_stats_debugging.csv file


distances_per_frame_df = calculate_distances_per_frame(Spot_stats) # Call the function to calculate distances_per_frame_df
absolute_directions_per_frame_df = calculate_absolute_directions_per_frame(Spot_stats) # Call the function to calculate directions_per_frame_df
weighted_directions_per_frame = calculate_weighted_directions_per_frame(Spot_stats) # Call the function tp calculate weighted_directions_per_frame
speeds_per_frame = calculate_speed(Spot_stats, 'POSITION_T')

Frame_stats_dfs = [distances_per_frame_df, absolute_directions_per_frame_df, weighted_directions_per_frame, speeds_per_frame]

Frame_stats = merge_dfs(Frame_stats_dfs, on='POSITION_T')
Frame_stats = Frame_stats.fillna(0)
Frame_stats.to_csv('Frame_stats.csv', index=False) # Save the Frame_stats created DataFrame into a newly created Frame_stats_debugging.csv file


