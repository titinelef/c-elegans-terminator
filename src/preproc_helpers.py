import numpy as np
import pandas as pd
import os
import re
import argparse
from tqdm import tqdm

# Constants (try changing to find best ones)
SEGMENT_LENGTH = 900
GAP_INTERPOLATION_LIMIT = 6
LONG_GAP_THRESHOLD = 7
MAX_SPEED_THRESHOLD = 10.0

def cap_extreme_speeds(df, max_speed=MAX_SPEED_THRESHOLD):
    """Cap extreme speed values to prevent tracking artifacts from skewing analysis."""
    df = df.copy()
    df['Speed'] = df['Speed'].clip(upper=max_speed)
    return df

def extract_filename_pattern(filename):
    """Extract the canonical filename pattern used in lifespan metadata.

    Args:
        filename: name of the raw CSV file.

    Returns:
        str | None: A pattern like `/<YYYYMMDD>_piworm<NN>_<recording>` if the
        filename matches the expected convention, otherwise None.
    """
    
    
    if 'coordinates_highestspeed_' in filename:
        match = re.search(r'coordinates_highestspeed_(\d{8})_(\d+)_(\d+)', filename)
        if match:
            date, worm_num, recording_num = match.groups()
            worm_num_padded = worm_num.zfill(2)
            piworm = f'piworm{worm_num_padded}'
            return f'/{date}_{piworm}_{recording_num}'
    
    return None

def trim_after_death(df, lifespan_df):
    """remove all rows after frame of death for a worm"""
    worm_pattern = extract_filename_pattern(df["source_file"].iloc[0])

    if worm_pattern is None:
        raise ValueError(f"Could not extract pattern from {df['source_file'].iloc[0]}")
    # Get death frame
    death_frame = lifespan_df.loc[lifespan_df["Filename"] == worm_pattern] 
    death_frame = int(death_frame.iloc[0]["LifespanInFrames"]) + 1 # keep death frame like other project
    
    # Trim
    df_trim = df[df["GlobalFrame"] <= death_frame].reset_index(drop=True)
    return df_trim

def split_into_segments(df, segment_length = SEGMENT_LENGTH):
    """Split a trajectory dataframe into a list of per-segment dataframes.


    Args:
        df: DataFrame containing a `segment` column.
        segment_length

    Returns:
        list[pandas.DataFrame]: One dataframe per unique segment with a
        `segment_index` column added.
    """
    df['Segment'] = (df['GlobalFrame'] - 2 ) // SEGMENT_LENGTH # all size SEGMENT_LENGTH except last one
    segments = []
    for segment_id in sorted(df['Segment'].unique()):
        segment_df = df[df['Segment'] == segment_id].copy()
        segment_df['Segment_index'] = segment_id
        segments.append(segment_df)
    
    return segments

def clean_segment_gaps(segment_df):
    """Repair short gaps by interpolation and remove long gaps.

    Args:
        segment_df: DataFrame of a single segment containing columns `x`, `y`,
            and `speed` where gaps may be NaN.

    Returns:
        pandas.DataFrame: Cleaned segment dataframe with short gaps interpolated
        and rows from long gaps removed.
    """
    gap_mask = segment_df['X'].isna()
    is_nan = gap_mask.astype(int)
    starts = (is_nan.diff() == 1).astype(int)
    if len(is_nan) > 0 and is_nan.iloc[0] == 1:
        starts.iloc[0] = 1
        
    gap_ids = starts.cumsum() * is_nan
    
    rows_to_remove = []
    for gap_id in gap_ids[gap_ids > 0].unique():
        indices = segment_df.index[gap_ids == gap_id].tolist()
        gap_size = len(indices)
        
        if gap_size <= GAP_INTERPOLATION_LIMIT:
            if len(indices) > 0:
                start_idx = max(segment_df.index.min(), indices[0] - 1)
                end_idx = min(segment_df.index.max(), indices[-1] + 1)
                segment_df.loc[start_idx:end_idx, ['X', 'Y', 'Speed']] = segment_df.loc[start_idx:end_idx, ['X', 'Y', 'Speed']].interpolate(method='linear')
        elif gap_size >= LONG_GAP_THRESHOLD:
            rows_to_remove.extend(indices)

    if rows_to_remove:
        segment_df = segment_df.drop(index=rows_to_remove).reset_index(drop=True)
    
    return segment_df

def calculate_turning_angle(df):
    """Compute per-frame turning angle in degrees.

    Args:
        df: DataFrame with at least `x` and `y` columns.

    Returns:
        pandas.DataFrame: Same dataframe with a `turning_angle` column in degrees
        added (first and last frames set to 0 if available).
    """
    if len(df) < 3 or 'X' not in df.columns or 'Y' not in df.columns:
        df['turning_angle'] = 0
        return df
    
    dx = df['X'].diff().values
    dy = df['Y'].diff().values
    
    angle = np.arctan2(dy, dx)
    angle_next = np.roll(angle, -1)
    turning_angle_rad = angle_next - angle
    turning_angle_rad = (turning_angle_rad + np.pi) % (2 * np.pi) - np.pi
    
    df['turning_angle'] = np.degrees(turning_angle_rad)
    df.loc[0, 'turning_angle'] = 0
    df.loc[df.index[-1], 'turning_angle'] = 0
    return df

def normalize_trajectory_data(df):
    """Normalize coordinates to [0,1], turning angles to [-1,1]"""
    df_normalized = df.copy()
    
    # Global coordinate bounds from the dataset
    XY_MIN = 0.0
    XY_MAX = 749.0
    
    # Global turning angle bounds
    ANGLE_MIN = -180.0
    ANGLE_MAX = 180.0
    
    if 'X' in df.columns and 'Y' in df.columns:
        df_normalized['X'] = (df['X'] - XY_MIN) / (XY_MAX - XY_MIN)
        df_normalized['Y'] = (df['Y'] - XY_MIN) / (XY_MAX - XY_MIN)
    
    if 'turning_angle' in df.columns:
        df_normalized['turning_angle'] = df['turning_angle'] / 180.0

    if 'Speed' in df.columns:
        speed_mean = np.mean(df['Speed'])
        speed_std = np.std(df['Speed'])
        df_normalized['Speed'] = (df['Speed'] - speed_mean) / speed_std
    
    return df_normalized

def rename_worm_id(df):
    """ 
    Extract canonical worm_id from the source_file column and store it as a new column. 
    """
    df_renamed =df.copy()
    worm_ids = []

    for filename in df_renamed['source_file'] :

        # convert to string to avoid 'float not iterable' errors
        filename_str = str(filename)

        if 'coordinates_highestspeed_' in filename_str:
            match = re.search(r'coordinates_highestspeed_(\d{8})_(\d+)_(\d+)', filename_str)

            if match:
                date, worm_num, recording_num = match.groups()
                worm_num_padded = worm_num.zfill(2)
                piworm = f'piworm{worm_num_padded}'
                worm_id =f'{date}_{piworm}_{recording_num}'
                worm_ids.append(worm_id)
            else :
                worm_ids.append(None)
        else :
            worm_ids.append(None)

    df_renamed["worm_id"] = worm_ids
    return df_renamed

def fill_nans_with_next_value(df):
    """
    Replace NaNs in each column with the next valid value in that column.
    
    Args:
        features: worm dataframe
    
    Returns:
        cleaned_2: same shape, NaNs replaced
    """
    cleaned = df.copy()

    # Skip empty or 1-frame segments (can't compute speed or turning angle)
    if len(cleaned) < 2:
        return None

    # Fill NaNs using backward then forward fill
    cleaned = cleaned.bfill().ffill()

    # Still some NaNs left? Fill numeric with 0
    for col in ["X", "Y", "Speed", "turning_angle"]:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].fillna(0)

    return cleaned

def save_worm_csv(df, filename, output_dir):
    """Save a worm dataframe into the output folder"""
    
    
    # Build full path
    out_path = os.path.join(output_dir, filename)

    # Save
    df.to_csv(out_path, index=False)