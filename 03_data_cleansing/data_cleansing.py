import pandas as pd


# path to the input file:
input_file = r"Z:\Shared\bryjalab\users\Branislav\Collagen Migration Assay DATA\data 23-7-24\run1\position_4!\C2-position_spots.csv"

df = pd.read_csv(input_file)


# Definition of unneccesary float columns in the df which are to be convertet to integers
unneccessary_float_columns = [
    'ID', 
    'Track ID', 
    'Time point', 
    'FRAME'
    ]

# Defining a function for cleansing of the raw dataframe
def butter(df, float_columns):

    # Loads the data into a DataFrame
    df = pd.DataFrame(df)

    # Reset the df index
    df = df.reset_index(drop=True)

    # Converts non-numeric values in selected columns to numeric values
    df = df.apply(pd.to_numeric, errors='coerce').dropna(subset=['X coordinate', 'Y coordinate', 'Time point'])

    # Sorts the data in the DataFrame by Track ID and Time point (time position)
    df = df.sort_values(by=['Track ID', 'Time point'])    

    # For some reason, the y coordinates extracted from trackmate are mirrored. That ofcourse would not affect the statistical tests, only the data visualization. However, to not get mindfucked..
    # Reflect y-coordinates around the midpoint for the directionality to be accurate, according to the microscope videos.
    y_mid = (df['Y coordinate'].min() + df['Y coordinate'].max()) / 2
    df['Y coordinate'] = 2 * y_mid - df['Y coordinate']

    # The dataset itself has a very chaotic, multirow column "title system". Therefore in this list are again defined columns, which from now on will be used for consistency.
    df.columns = [
        'LABEL', 
        'ID', 
        'Track ID', 
        'QUALITY', 
        'X coordinate', 
        'Y coordinate', 
        'POSITION_Z', 
        'Time point', 
        'FRAME', 
        'RADIUS', 
        'VISIBILITY', 
        'MANUAL_SPOT_COLOR', 
        'MEAN_INTENSITY_CH1', 
        'MEDIAN_INTENSITY_CH1', 
        'MIN_INTENSITY_CH1', 
        'MAX_INTENSITY_CH1', 
        'TOTAL_INTENSITY_CH1', 
        'STD_INTENSITY_CH1', 
        'EXTRACK_P_STUCK', 
        'EXTRACK_P_DIFFUSIVE', 
        'CONTRAST_CH1', 
        'SNR_CH1'
        ]

    # Droping all non numeric values, also dropping whole columns only containing non-numeric values.
    df = df.dropna(axis=1)

    # Here we convert the unnecessary floats (from the list in which we defined them) to integers
    df[float_columns] = df[float_columns].astype(int)

    return df
# Executing the function the function 
df = butter(df, unneccessary_float_columns)

# Saving the cleansed DataFrame to a new csv file	
df.to_csv("buttered.csv", index=False)