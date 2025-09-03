import encodings
from os import path
import numpy as np
import pandas as pd
from math import log10, floor, ceil
import os.path as op
from typing import List, Any, Callable, Literal, Optional, Union
from pandas.api.types import is_object_dtype



def _pick_encoding(path, encodings=("utf-8", "cp1252", "latin1", "iso8859_15")):
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False if enc != "utf-8" else True)
        except UnicodeDecodeError:
            continue

def _has_strings(s: pd.Series) -> bool:
    # pandas "string" dtype (pyarrow/python)
    if isinstance(s.dtype, pd.StringDtype):
        return s.notna().any()
    # categorical of strings?
    if isinstance(s.dtype, pd.CategoricalDtype):
        return isinstance(s.dtype.categories.dtype, pd.StringDtype) and s.notna().any()
    # numeric, datetime, bool, etc.
    if not is_object_dtype(s.dtype):
        return False
    # Fallback for object-dtype (mixed types): minimal Python loop over NumPy array
    arr = s.to_numpy(dtype=object, copy=False)
    return any(isinstance(v, (str, np.str_)) for v in arr)



class DataLoader:
    

    @staticmethod
    def GetDataFrame(filepath: str) -> pd.DataFrame:
        """
        Loads a DataFrame from a file based on its extension.
        Supported formats: CSV, Excel, Feather, Parquet, HDF5, JSON.
        """
        _, ext = op.splitext(filepath.lower())

        try:
            if ext == '.csv':
                return _pick_encoding(filepath)
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
        except ValueError as e:
            raise e(f"{ext} is not a supported file format.")
        except Exception as e:
            raise e(f"Failed to load file '{filepath}': {e}")
    

    @staticmethod
    def Extract(df: pd.DataFrame, id_col: str, t_col: str, x_col: str, y_col: str, mirror_y: bool = True) -> pd.DataFrame:
        # Keep only relevant columns and convert to numeric
        df = df[[id_col, t_col, x_col, y_col]].apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)

        # Mirror Y if needed
        if mirror_y:
            """
            TrackMate may export y-coordinates mirrored.
            This does not affect statistics but leads to incorrect visualization.
            Here Y data is reflected across its midpoint for accurate directionality.
            """
            y_mid = (df[y_col].min() + df[y_col].max()) / 2
            df[y_col] = 2 * y_mid - df[y_col]

        # Standardize column names
        return df.rename(columns={id_col: 'Track ID', t_col: 'Time point', x_col: 'X coordinate', y_col: 'Y coordinate'})


    @staticmethod
    def GetColumns(path: str) -> List[str]:
        """
        Returns a list of column names from the DataFrame.
        """
        df = DataLoader.GetDataFrame(path)  # or pd.read_excel(path), depending on file type
        return df.columns.tolist()
    
    @staticmethod
    def FindMatchingColumn(columns: List[str], lookfor: List[str]) -> str:
        """
        Looks for matches with any of the provided strings.
        - First tries exact matches.
        - Then checks if the column starts with any of given terms.
        - Finally checks if any term is a substring of the column name.
        If no match is found, returns None.
        """

        # Normalize columns for matching
        normalized_columns = [
            (col, str(col).replace('_', ' ').strip().lower() if col is not None else '') for col in columns
        ]
        # Try exact matches first
        for col, norm_col in normalized_columns:
            for look in lookfor:
                if norm_col == look.lower():
                    return col
        # Then try startswith
        for col, norm_col in normalized_columns:
            for look in lookfor:
                if norm_col.startswith(look.lower()):
                    return col
        # Then try substring
        for col, norm_col in normalized_columns:
            for look in lookfor:
                if look.lower() in norm_col:
                    return col
        return None
        



class Process:

    @staticmethod
    def TryConvertNumeric(x: Any) -> Any:
        """
        Try to convert a string to an int or float, otherwise return the original value.
        """
        try:
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
        
    @staticmethod
    def TryFloat(x: Any) -> Any:
        """
        Try to convert a string to an int or float, otherwise return the original value.
        """
        try:
            if isinstance(x, str):
                x_stripped = x.strip()
                num = float(x_stripped)
                if num.is_integer():
                    return float(num)
                else:
                    return num
            else:
                return x
        except ValueError:
            return x

    @staticmethod
    def MergeDFs(dataframes: List[pd.DataFrame], on: List[str]) -> pd.DataFrame:
        """
        Merges a list of DataFrames on the specified columns using an outer join.
        All values are coerced to string before merging, then converted back to numeric where possible.

        Parameters:
            dataframes: List of DataFrames to merge.
            on: List of column names to merge on.

        Returns:
            pd.DataFrame: The merged DataFrame with numerics restored where possible.
        """
        if not dataframes:
            raise ValueError("No dataframes provided for merging.")

        # Initialize the first DataFrame as the base for merging (all values as string)
        merged_df = dataframes[0].map(str)

        for df in dataframes[1:]:
            df = df.map(str)
            # Ensure all key columns are present for merging
            merge_columns = [col for col in df.columns if col not in merged_df.columns or col in on]
            merged_df = pd.merge(
                merged_df,
                df[merge_columns],
                on=on,
                how='outer'
            )

        # Use the static method for numeric conversion
        merged_df = merged_df.applymap(Process.TryConvertNumeric)
        return merged_df


    def Round(value, step, round_method="nearest"):
        """
        Rounds value to the nearest multiple of step.
        """
        if round_method == "nearest":
            return round(value / step) * step
        elif round_method == "floor":
            return floor(value / step) * step
        elif round_method == "ceil":
            return ceil(value / step) * step
        else:
            raise ValueError(f"Unknown round method: {round_method}")



class Calc:
    
    @staticmethod
    def Spots(df: pd.DataFrame) -> pd.DataFrame:

        """
        Compute per-frame tracking metrics for each cell track in the DataFrame:
        - Distance: Euclidean distance between consecutive positions
        - Direction (rad): direction of travel in radians
        - Track length: cumulative distance along the track
        - Net distance: straight-line distance from track start
        - Confinement ratio: Net distance / Track length

        Expects columns: Condition, Replicate, Track ID, X coordinate, Y coordinate, Time point
        Returns a DataFrame sorted by Condition, Replicate, Track ID, Time point with new metric columns.
        """
        if df.empty:
            return df.copy()

        df.sort_values(by=['Condition', 'Replicate', 'Track ID', 'Time point'], inplace=True)

        # Sort and work on a copy
        # df = df.sort_values(['Condition', 'Replicate', 'Track ID', 'Time point']).copy()
        grp = df.groupby(['Condition', 'Replicate', 'Track ID'], sort=False)

        # Distance between current and next position
        df['Distance'] = np.sqrt(
            (grp['X coordinate'].shift(-1) - df['X coordinate'])**2 +
            (grp['Y coordinate'].shift(-1) - df['Y coordinate'])**2
            ).fillna(0)

        # Direction of travel (radians) based on diff to previous point
        df['Direction (rad)'] = np.arctan2(
            grp['Y coordinate'].diff(),
            grp['X coordinate'].diff()
            ).fillna(0)

        # Cumulative track length
        df['Track length'] = grp['Distance'].cumsum()

        # Net (straight-line) distance from the start of the track
        start = grp[['X coordinate', 'Y coordinate']].transform('first')
        df['Net distance'] = np.sqrt(
            (df['X coordinate'] - start['X coordinate'])**2 +
            (df['Y coordinate'] - start['Y coordinate'])**2
            )

        # Confinement ratio: net distance vs. actual path length
        # Avoid division by zero by replacing zeros with NaN, then fill
        df['Confinement ratio'] = (df['Net distance'] / df['Track length'].replace(0, np.nan)).fillna(0)

        return df

    @staticmethod
    def Tracks(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute comprehensive track-level metrics for each cell track in the DataFrame, including:
        - Track length: sum of Distance
        - Net distance: straight-line from first to last position
        - Confinement ratio: Net distance / Track length
        - Min speed, Max speed, Mean speed, Std speed, Median speed (per-track on Distance)
        - Mean direction (rad/deg), Std deviation (rad/deg), Median direction (rad/deg) (circular stats)

        Expects columns: Condition, Replicate, Track ID, Distance, X coordinate, Y coordinate, Direction (rad)
        Returns a single DataFrame indexed by Condition, Replicate, Track ID with all metrics.
        """
        if df.empty:
            cols = [
                'Condition','Replicate','Track ID',
                'Track length','Net distance','Confinement ratio',
                'Speed min','Speed max','Speed mean','Speed std','Speed median',
                'Direction mean (rad)','Direction std (rad)','Direction median (rad)',
                'Direction mean (deg)','Direction std (deg)','Direction median (deg)'
            ]
            return pd.DataFrame(columns=cols)

        # Group by track
        grp = df.groupby(['Condition','Replicate','Track ID'], sort=False)

        agg = grp.agg(
            **{
                'Track length': ('Distance', 'sum'),
                'Speed mean':  ('Distance', 'mean'),
                'Speed std':   ('Distance', 'std'),
                'start_x':     ('X coordinate', 'first'),
                'end_x':       ('X coordinate', 'last'),
                'start_y':     ('Y coordinate', 'first'),
                'end_y':       ('Y coordinate', 'last')
            }
        )

        # Compute net displacement and confinement ratio
        agg['Net distance'] = np.hypot(agg['end_x'] - agg['start_x'], agg['end_y'] - agg['start_y'])
        agg['Confinement ratio'] = (agg['Net distance'] / agg['Track length'].replace(0, np.nan)).fillna(0)
        agg = agg.drop(columns=['start_x','end_x','start_y','end_y'])

        # Circular direction statistics: need sin & cos per observation
        sin_cos = df.assign(_sin=np.sin(df['Direction (rad)']), _cos=np.cos(df['Direction (rad)']))
        dir_agg = sin_cos.groupby(['Condition','Replicate','Track ID'], sort=False).agg(
            mean_sin=('_sin','mean'), mean_cos=('_cos','mean'),
            median_sin=('_sin','median'), median_cos=('_cos','median')
        )
        # derive circular metrics
        dir_agg['Direction mean (rad)'] = np.arctan2(dir_agg['mean_sin'], dir_agg['mean_cos'])
        dir_agg['Direction std (rad)'] = np.hypot(dir_agg['mean_sin'], dir_agg['mean_cos'])
        dir_agg['Direction median (rad)'] = np.arctan2(dir_agg['median_sin'], dir_agg['median_cos'])
        dir_agg['Direction mean (deg)'] = np.degrees(dir_agg['Direction mean (rad)']) % 360
        dir_agg['Direction std (deg)'] = np.degrees(dir_agg['Direction std (rad)']) % 360
        dir_agg['Direction median (deg)'] = np.degrees(dir_agg['Direction median (rad)']) % 360
        dir_agg = dir_agg.drop(columns=['mean_sin','mean_cos','median_sin','median_cos'])

            # Count points per track
        # number of rows (frames) per track
        point_counts = grp.size().rename('Track points')

        # Merge all metrics into one DataFrame
        result = agg.merge(dir_agg, left_index=True, right_index=True)
        # Merge point counts
        result = result.merge(point_counts, left_index=True, right_index=True)
        result = result.reset_index()
        return result

    @staticmethod
    def Time(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-frame (time point) summary metrics grouped by Condition, Replicate, Time point:
        - Track length, Net distance, Confinement ratio distributions: min, max, mean, std, median
        - Speed (Distance) distributions as Speed min, Speed max, Speed mean, Speed std, Speed median
        - Direction (rad) distributions (circular): Direction mean (rad), Direction std (rad), Direction median (rad)
            and corresponding degrees

        Expects columns: Condition, Replicate, Time point, Track length, Net distance,
                        Confinement ratio, Distance, Direction (rad)
        Returns a DataFrame indexed by Condition, Replicate, Time point with all time-point metrics.
        """
        if df.empty:
            # define columns
            cols = ['Condition','Replicate','Time point'] + \
                [f'{metric} {stat}' for metric in ['Track length','Net distance','Confinement ratio'] for stat in ['min','max','mean','std','median']] + \
                [f'Speed {stat}' for stat in ['min','max','mean','std','median']] + \
                ['Direction mean (rad)','Direction std (rad)','Direction median (rad)',
                    'Direction mean (deg)','Direction std (deg)','Direction median (deg)']
            return pd.DataFrame(columns=cols)

        group_cols = ['Condition','Replicate','Time point']

        # 1) stats on track metrics per frame
        metrics = ['Track length','Net distance','Confinement ratio']
        agg_funcs = ['min','max','mean','std','median']
        # build agg dict
        agg_dict = {m: agg_funcs for m in metrics}
        frame_agg = df.groupby(group_cols).agg(agg_dict)
        # flatten columns
        frame_agg.columns = [f'{metric} {stat}' for metric, stat in frame_agg.columns]

        # 2) speed stats (Distance distributions)
        speed_agg = df.groupby(group_cols)['Distance'].agg(['min','max','mean','std','median'])
        speed_agg.columns = [f'Speed {stat}' for stat in speed_agg.columns]

        # 3) circular direction stats per frame
        # compute sin/cos columns
        tmp = df.assign(_sin=np.sin(df['Direction (rad)']), _cos=np.cos(df['Direction (rad)']))
        dir_frame = tmp.groupby(group_cols).agg({'_sin':'mean','_cos':'mean','Direction (rad)':'count'})
        # mean direction
        dir_frame['Direction mean (rad)'] = np.arctan2(dir_frame['_sin'], dir_frame['_cos'])
        # circular std: R = sqrt(mean_sin^2+mean_cos^2)
        dir_frame['Direction std (rad)'] = np.hypot(dir_frame['_sin'], dir_frame['_cos'])
        # median direction: use groupby apply median sin/cos
        median = tmp.groupby(group_cols).agg({'_sin':'median','_cos':'median'})
        dir_frame['Direction median (rad)'] = np.arctan2(median['_sin'], median['_cos'])
        # degrees
        dir_frame['Direction mean (deg)'] = np.degrees(dir_frame['Direction mean (rad)']) % 360
        dir_frame['Direction std (deg)'] = np.degrees(dir_frame['Direction std (rad)']) % 360
        dir_frame['Direction median (deg)'] = np.degrees(dir_frame['Direction median (rad)']) % 360
        dir_frame = dir_frame.drop(columns=['_sin','_cos','Direction (rad)'], errors='ignore')

        # merge all
        time_stats = frame_agg.merge(speed_agg, left_index=True, right_index=True)
        time_stats = time_stats.merge(dir_frame, left_index=True, right_index=True)
        time_stats = time_stats.reset_index()

        return time_stats



class Threshold:

    @staticmethod
    def Normalize_01(df, col) -> pd.Series:
        """
        Normalize a column to the [0, 1] range.
        """
        # s = pd.to_numeric(df[col], errors='coerce')
        try:
            s = pd.Series(Process.TryFloat(df[col]), dtype=float)
            if _has_strings(s):
                normalized = pd.Series(0.0, index=s.index, name=col)
            lo, hi = s.min(), s.max()
            if lo == hi:
                normalized = pd.Series(0.0, index=s.index, name=col)
            else:
                normalized = pd.Series((s - lo) / (hi - lo), index=s.index, name=col)

        except Exception:
            normalized = pd.Series(0.0, index=df.index, name=col)

        return normalized  # <-- keeps index

    @staticmethod
    def JoinByIndex(a: pd.Series, b: pd.Series) -> pd.DataFrame:
        """
        Join two Series of potentially different lengths into a DataFrame.
        """

        if b.index.is_unique and not a.index.is_unique:
            df = a.rename(a.name).to_frame().set_index(a.index)
            df[b.name] = b.reindex(df.index)
        else:
            df = b.rename(b.name).to_frame().set_index(b.index)
            df[a.name] = a.reindex(df.index)

        return df
    
