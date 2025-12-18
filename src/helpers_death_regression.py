import os
import glob
import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def prepare_X_y(
    segments: List[Dict[str, Any]], 
    feature_names: List[str], 
    feature_mapping: Dict[str, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepares the X, y datasets and the mask array from a list of segments.

    Args:
        segments: List of segment dictionaries (containing 'features', 'y', 'mask').
        feature_names: List of feature column names to use (e.g., ['Speed', 'Turning_Angle']).
        feature_mapping: Dictionary mapping {column_name: index} to extract data.

    Returns:
        A tuple (X, y, mask) ready for model training.
    """
    
    try:
        selected_indices = [feature_mapping[name] for name in feature_names]
    except KeyError as e:
        raise ValueError(f"Feature name '{e}' not found in the feature mapping.")
        
    # 2. Extract the data
    # For each segment, take the 'features' array and select the columns 
    # corresponding to the chosen indices.
    X = np.array([seg['features'][:, selected_indices] for seg in segments])
    y = np.array([seg['y'] for seg in segments])
    mask = np.array([seg['mask'] for seg in segments])
    
    return X, y, mask


def diagnose_segments(segment_list, label=""):
    nan_flags = [np.isnan(seg['features']).any() for seg in segment_list]
    inf_flags = [np.isinf(seg['features']).any() for seg in segment_list]
    both_flags = [n and i for n, i in zip(nan_flags, inf_flags)]

    print(f"\n--- Diagnostics {label} ---")
    print("Segments total :", len(segment_list))
    print("Contain NaN    :", sum(nan_flags))
    print("Contain inf    :", sum(inf_flags))
    print("Contain both   :", sum(both_flags))
    return nan_flags, inf_flags, both_flags



def load_data_death_regressor(
    data_dir: str,
    segments_subdir: str,
    lifespan_file: str,
    segment_length: int,
    feature_cols: List[str],
    debug_mode: bool = False
) -> Dict[str, Any]:
    """
    Loads, processes, and prepares time-series segments for a death regression model.

    The function reads lifespan data and time-series segments, filters invalid 
    segments, standardizes segment length via padding/truncation, and computes 
    the time-to-death target variable.

    Args:
        data_dir: Base directory containing all data.
        segments_subdir: Subdirectory within data_dir where segment CSVs are located.
        lifespan_file: Name of the lifespan summary CSV file within data_dir.
        segment_length: The fixed length to which all segments will be padded or truncated.
        feature_cols: List of columns to use as input features.
        debug_mode: If True, enables detailed log messages.

    Returns:
        A dictionary containing the list of accepted segments, rejection counts, 
        and the total number of segment files found.
    """
    
    DEBUG = debug_mode

    def log_debug(*args):
        """Prints debug messages if the global DEBUG flag is set to True."""
        if DEBUG:
            print("[DEBUG]", *args)
            
    # --- 1. Define Paths ---
    SEGMENTS_DIR = os.path.join(data_dir, segments_subdir)
    LIFESPAN_FILE_PATH = os.path.join(data_dir, lifespan_file)

    log_debug("--- Starting Data Loader ---")
    log_debug(f"Data Directory: {data_dir}")
    log_debug(f"Segments Directory: {SEGMENTS_DIR}")
    log_debug(f"Lifespan File Path: {LIFESPAN_FILE_PATH}")
    log_debug(f"Target Segment Length: {segment_length}")
    log_debug(f"Feature Columns: {feature_cols}")

    # --- 2. Load Lifespan Data ---
    try:
        lifespan_df = pd.read_csv(LIFESPAN_FILE_PATH)
    except Exception as e:
        print(f"FATAL ERROR: Could not read {LIFESPAN_FILE_PATH}: {e}")
        raise SystemExit(1)

    # Standardize column names
    lifespan_df.columns = lifespan_df.columns.str.strip()
    colmap = {c.lower(): c for c in lifespan_df.columns}

    # Identify essential columns (filename and lifespan frames)
    filename_col = colmap.get('filename') or colmap.get('file_name') or colmap.get('files') or colmap.get('source_file') or colmap.get('file')
    life_col = colmap.get('lifespaninframes') or colmap.get('lifespan') or colmap.get('lifespan_in_frames') or colmap.get('lifespanframes') or colmap.get('total_frames')

    if filename_col is None or life_col is None:
        print("Expected columns (filename or lifespan in frames) not found in lifespan summary CSV. Available columns:")
        print(list(lifespan_df.columns))
        raise SystemExit(1)

    # Prepare lifespan map
    lifespan_df['Worm_ID_raw'] = lifespan_df[filename_col].astype(str)
    # Extract ID by removing extension and leading slashes
    lifespan_df['Worm_ID'] = lifespan_df['Worm_ID_raw'].apply(lambda x: os.path.splitext(x.strip())[0].lstrip('/'))
    lifespan_map = lifespan_df.set_index('Worm_ID')[life_col].to_dict()

    print(f"Loaded {len(lifespan_df)} lifespan entries. Example keys: {list(lifespan_map.keys())[:5]}")
    print("-" * 60)

    # --- 3. Process Segment Files ---
    segment_files = glob.glob(os.path.join(SEGMENTS_DIR, "*.csv"))
    all_segments: List[Dict[str, Any]] = []
    
    # Rejection reasons counter
    rejection_counts = Counter({
        'No_Lifespan_Match': 0,
        'Missing_Features': 0,
        'Missing_Worm_ID': 0, # Added for clarity
        'After_Death': 0,
        'Pattern_Mismatch': 0, # Kept for generality, though not strictly used below
        'Read_Error': 0, 
        'Empty_Segment': 0
    })

    for file in segment_files:
        log_debug("\n--------------------------------------")
        log_debug(f"🔍 Checking file: {file}")

        # Load CSV
        try:
            df = pd.read_csv(file)
            log_debug("CSV loaded successfully")
        except Exception as e:
            rejection_counts['Read_Error'] += 1
            log_debug("Could not read CSV:", e)
            continue

        # Check for essential segment ID column
        if "worm_id" not in df.columns:
            rejection_counts['Missing_Worm_ID'] += 1
            log_debug("Missing 'worm_id' column → reject")
            continue

        worm_id_candidate = str(df["worm_id"].iloc[0]).strip()
        log_debug("Worm ID extracted:", worm_id_candidate)

        # Lifespan lookup
        total_lifespan: Optional[float] = lifespan_map.get(worm_id_candidate)

        if total_lifespan is None:
            rejection_counts['No_Lifespan_Match'] += 1
            log_debug(f"No lifespan match for '{worm_id_candidate}' → reject")
            continue

        log_debug("Lifespan found:", total_lifespan)

        # Feature verification
        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            rejection_counts['Missing_Features'] += 1
            log_debug("Missing features:", missing_features)
            continue

        # Frames / Length check
        original_length = df.shape[0]
        if original_length == 0:
            rejection_counts['Empty_Segment'] += 1
            log_debug("Empty segment dataframe → reject")
            continue
            
        try:
            segment_start_frame = int(df['GlobalFrame'].iloc[0])
        except Exception:
            rejection_counts['Read_Error'] += 1 # Frame column issue treated as read error
            log_debug("Could not determine segment start frame → reject")
            continue

        # --- 4. Feature Extraction & Padding ---
        features_raw = df[feature_cols].values.astype(np.float32)
        current_length = features_raw.shape[0]
        used_length = min(current_length, segment_length)
        
        features: np.ndarray
        
        if current_length > segment_length:
            # Truncate
            features = features_raw[:segment_length, :]
            log_debug(f"Segment truncated from {current_length} to {segment_length}")
        else:
            # Pad
            padding_needed = segment_length - current_length
            features = np.pad(features_raw, ((0, padding_needed), (0, 0)),
                              mode='constant', constant_values=0.0)
            log_debug(f"Segment padded from {current_length} to {segment_length}")

        # Mask generation (1.0 for valid data, 0.0 for padding)
        mask = np.zeros((segment_length,), dtype=np.float32)
        mask[:used_length] = 1.0

        # --- 5. Target Calculation (Time-to-death) ---
        
        # Calculate the last frame of the *actual* data used in the segment
        last_frame_observed = segment_start_frame + used_length - 1
        
        # Check for segments starting or entirely occurring after death
        if last_frame_observed > total_lifespan:
            # The original code did not reject but calculated pct_life_remaining based on the last observed frame.
            # However, for a death regression model, segments entirely after death 
            # might introduce noise or should be explicitly handled/filtered.
            # Assuming the intent is to filter segments where the *observation* # goes past the death frame:
            if segment_start_frame > total_lifespan:
                 rejection_counts['After_Death'] += 1
                 log_debug(f"Segment starts at frame {segment_start_frame}, after death at {total_lifespan} → reject")
                 continue
            
            # If the segment crosses the death boundary, the target represents 0% life remaining
            pct_life_remaining = 0.0
            log_debug("Segment observed life beyond total lifespan, target set to 0.0")
        else:
            # Time remaining / Total lifespan
            pct_life_remaining = max(0.0, (total_lifespan - last_frame_observed) / total_lifespan)
        
        target = float(pct_life_remaining)
        log_debug(f"Segment Start Frame: {segment_start_frame}, Last Frame Observed: {last_frame_observed}")
        log_debug(f"Total Lifespan: {total_lifespan}, Target (Pct Life Remaining): {target:.4f}")

        # --- 6. Store Segment ---
        all_segments.append({
            'features': features,
            'mask': mask,
            'y': target,
            'worm_id': worm_id_candidate,
            'start_frame': segment_start_frame,
            'valid_length': used_length
        })

    # --- 7. Summary ---
    print("---- Segment Loading Summary ----")
    print(f"Segment files found: {len(segment_files)}")
    print(f"Segments accepted: {len(all_segments)}")
    print("Rejection counts:", dict(rejection_counts))
    if len(all_segments) == 0:
        print("ERROR: No usable segments remaining after filtering.")
        raise SystemExit(1)
    print("-" * 60)
    
    return {
        'segments': all_segments,
        'rejection_counts': rejection_counts,
        'total_files': len(segment_files)
    }

def plot_feature_and_target_distributions(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    feature_names_list: List[str]
):
    """
    Generates and displays histograms for input features (X) and the target variable (y), 
    along with corresponding descriptive statistics for the training set.

    Args:
        X_train: Training features array (shape: N_segments, Segment_Length, N_features).
        y_train: Training target array (shape: N_segments).
        feature_names_list: List of names corresponding to the features in X_train.
    """
    
    # --- 1. Feature Distribution (Histograms) ---
    
    # Flatten X_train from (N, T, F) to (N*T, F) for distribution analysis
    X_train_flat = X_train.reshape(-1, X_train.shape[2])
    data_df = pd.DataFrame(X_train_flat, columns=feature_names_list)
    num_features_to_plot = data_df.shape[1] 

    print("--- 1. Feature Distribution Histograms ---")
    plt.figure(figsize=(5 * num_features_to_plot, 5))
    plt.suptitle(
        'Distribution of Input Features (Across All Segments and Timesteps)', 
        fontsize=12
    )

    for i, feature_name in enumerate(feature_names_list):
        plt.subplot(1, num_features_to_plot, i + 1)
        
        # Histogram
        data_df[feature_name].hist(bins=50, edgecolor='black', alpha=0.7) 
        
        # Mean and Std on the whole dataset
        mean_val = data_df[feature_name].mean()
        std_val = data_df[feature_name].std()
        
        plt.title(f'{feature_name}')
        plt.xlabel(f'$\mu={mean_val:.3f}, \sigma={std_val:.3f}$')
        plt.ylabel('Frequency (Counts)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- 2. Feature Statistics ---

    # Calculate mean and standard deviation across all samples (axis 0) 
    # and all timesteps (axis 1)
    mean_per_feature = np.mean(X_train, axis=(0, 1))
    std_per_feature = np.std(X_train, axis=(0, 1))

    print("\n--- 2. Feature Statistics (X_train) ---")
    print(f"Features: {feature_names_list}")
    print("Mean per Feature (Across all time/segments):", mean_per_feature.round(4))
    print("Standard Deviation (Std):                   ", std_per_feature.round(4))

    # --- 3. Target Variable Distribution ---
    
    print("\n--- 3. Target Variable Statistics and Histogram (y_train) ---")

    plt.figure(figsize=(3, 2))
    plt.hist(y_train, bins=20, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Target Variable (y_train - Remaining Lifespan)')
    plt.xlabel(f'Target Value (e.g., Percentage of Lifespan Remaining)')
    plt.ylabel('Number of Segments')
    plt.grid(axis='y', alpha=0.5)
    plt.show()

    print(f"Mean of y_train:       {y_train.mean():.4f}")
    print(f"Standard Deviation (Std): {y_train.std():.4f}")
    print(f"Minimum value of y_train: {y_train.min():.4f}")
    print(f"Maximum value of y_train: {y_train.max():.4f}")

def evaluate_and_plot_results(model, X_test, y_test, history):
    """
    Evaluates the model on test data, prints R2 and MAE metrics, 
    and plots the learning curve as well as the prediction scatter plot.

    Args:
        model: The trained Keras/TensorFlow model.
        X_test (np.array): Test data features.
        y_test (np.array): True target values for test data.
        history: The history object returned by model.fit().
    """
    
    # 1. Predictions
    # Use .predict() and .flatten() to get a 1D array of predictions
    y_pred = model.predict(X_test).flatten()
    
    # Ensure true values are also 1D for metric calculation
    y_test_flat = y_test.flatten()

    # 2. Calculate metrics
    r2 = r2_score(y_test_flat, y_pred)
    mae = mean_absolute_error(y_test_flat, y_pred)

    # 3. Print metrics
    print(f"--- RESULTS ---")
    print(f"R2 Score : {r2:.4f}")
    print(f"MAE      : {mae:.4f}")

    plt.figure(figsize=(14, 6)) 

    plt.subplot(1, 2, 1)
    if 'loss' in history.history and 'val_loss' in history.history:
        plt.plot(history.history['loss'], label='Train MSE')
        plt.plot(history.history['val_loss'], label='Val MSE')
        plt.title('Learning Curve')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True)
    else:
        print("\nWarning: 'loss' or 'val_loss' not found in history.history. Cannot plot learning curve.")

    plt.subplot(1, 2, 2)
    plt.scatter(y_test_flat * 100, y_pred * 100, alpha=0.3, s=15, label='Segments')
    max_val = max(np.max(y_test_flat * 100), np.max(y_pred * 100))
    min_val = min(np.min(y_test_flat * 100), np.min(y_pred * 100))
    
    # Ensure scale is centered around [0, 100]
    plot_min = max(0, min_val - 5)
    plot_max = min(100, max_val + 5)
    
    plt.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', label='Perfect Pred')

    plt.title(f'Prediction Accuracy (R2={r2:.2f})')
    plt.xlabel('True Life % Remaining')
    plt.ylabel('Predicted Life % Remaining')

    plt.xlim(0, 105)
    plt.ylim(0, 105)    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()