import pandas as pd
import os


# Inputting files

input_file1 = r"C:\Users\modri\Desktop\Lab\runs from the same experiment\C2-position_spots1.csv"
input_file2 = r"C:\Users\modri\Desktop\Lab\runs from the same experiment\C2-position_spots2.csv"
input_file3 = r"C:\Users\modri\Desktop\Lab\runs from the same experiment\C2-position_spots3.csv"

input_filesA = [input_file1, input_file2]
input_filesB = [input_file1, input_file3]
input_filesC = [input_file2, input_file3]

parsed_input = [input_filesA, input_filesB, input_filesC]



def load_DataFrame(filepath: str) -> pd.DataFrame:
    """
    Loads a DataFrame from a file based on its extension.
    Supported formats: CSV, Excel, Feather, Parquet, HDF5, JSON.
    """

    _, ext = os.path.splitext(filepath.lower())

    try:
        if ext == '.csv':
            return pd.read_csv(filepath)
        elif ext in ['.xls', '.xlsx']:
            return pd.read_excel(filepath)
        elif ext == '.feather':
            return pd.read_feather(filepath)
        elif ext == '.parquet':
            return pd.read_parquet(filepath)
        elif ext in ['.h5', '.hdf5']:
            return pd.read_hdf(filepath)
        elif ext == '.json':
            return pd.read_json(filepath)
        else:
            raise ValueError(f"{ext} is not a supported file format.")
    except Exception as e:
        raise RuntimeError(f"Failed to load file '{filepath}': {e}")



def get_columns(df: pd.DataFrame) -> list:
    """
    Returns a list of column names from the DataFrame.
    """
    return df.columns.tolist()




def parsed_file(input_files):  # File-reading

    if input_files is None:
        return pd.DataFrame()

    all_data_dflt = []
    for list_count, sublist in enumerate(input_files, start=1):  # Enumerate and cycle through input lists
        condition = list_count  # Assign a unique condition number for each list
        for file_count, file_dflt in enumerate(sublist, start=1):  # Enumerate and cycle through files in the sublist
            df = load_DataFrame(file_dflt)


            df['Condition'] = condition  # Assign the condition number
            df['Replicate'] = file_count  # Assign the replicate number

            all_data_dflt.append(df)  # Store processed DataFrame

    default = pd.concat(all_data_dflt, axis=0)  # Join the DataFrames
    return default

# Example usage
buttered = parsed_file(parsed_input)  # Displayed data is the processed DataFrame