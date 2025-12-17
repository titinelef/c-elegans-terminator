
import os, random
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, GlobalAveragePooling1D, Dense, Dropout, MaxPooling1D
from tensorflow.keras.models import Model



def plot_trajectory(df, x_col="X", y_col="Y", title="Trajectory"):
    """
    Plot a 2D worm trajectory.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing X/Y coordinates.
    x_col, y_col : str
        Column names for x and y positions.
    title : str
        Plot title.
    """

    plt.figure(figsize=(6, 6))
    
    plt.plot(df[x_col], df[y_col], marker='o', markersize=2, linewidth=1)
    
    # Mark start and end
    plt.scatter(df[x_col].iloc[0], df[y_col].iloc[0], color='green', s=50, label="Start")
    plt.scatter(df[x_col].iloc[-1], df[y_col].iloc[-1], color='red', s=50, label="End")
    
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    plt.axis('equal')  # preserve trajectory geometry
    plt.grid(True)
    plt.show()



def recenter_coordinates(df, x_col="X", y_col="Y"):
    """
    Recenter trajectory so the first frame is at (0, 0).

    Parameters
    ----------
    df : pd.DataFrame
        Input trajectory.
    x_col, y_col : str
        Original coordinate columns.

    Returns
    -------
    pd.DataFrame
        Copy of df with added columns:
        - X_centered
        - Y_centered
    """
    df = df.copy()

    #first position becomes the origin
    x0, y0 = df[x_col].iloc[0], df[y_col].iloc[0]
    df["X_centered"] = df[x_col] - x0
    df["Y_centered"] = df[y_col] - y0

    return df

def rotate_to_initial_heading(df, x_col="X_centered", y_col="Y_centered"):
    """
    Rotate trajectory so the initial movement direction is aligned to 0 radians.

    Parameters
    ----------
    df : pd.DataFrame
        Centered trajectory.
    x_col, y_col : str
        Centered coordinate columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - X_rot
        - Y_rot
    """
    df = df.copy()

    # Compute initial displacement (need at least 2 frames)
    dx = df[x_col].iloc[1] - df[x_col].iloc[0]
    dy = df[y_col].iloc[1] - df[y_col].iloc[0]

    # Angle of initial heading
    theta = np.arctan2(dy, dx)

    # Rotation matrix to align with 0 radians
    c, s = np.cos(-theta), np.sin(-theta)
    R = np.array([[c, -s],
    [s, c]])

    # Apply rotation
    coords = df[[x_col, y_col]].values.T # shape (2, timesteps)
    rotated = R @ coords # matrix multiplication

    df["X_rot"] = rotated[0]
    df["Y_rot"] = rotated[1]

    return df


def load_and_preprocess(PATH):
    """
    Load all trajectory segments from a folder and preprocess them.

    Steps
    -----
    1. Load CSV files
    2. Recenter coordinates
    3. Rotate to initial heading
    4. Standardize per segment

    Parameters
    ----------
    PATH : str
        Directory containing segment CSV files.

    Returns
    -------
    list[pd.DataFrame]
        Preprocessed segment DataFrames.
    """

    all_segment_dfs = []
    seg_files = sorted(os.listdir(PATH))

    for fname in tqdm(seg_files):
        if not fname.endswith(".csv"):
            continue

        df = pd.read_csv(os.path.join(PATH, fname))
        all_segment_dfs.append(df)

    centered_dfs = [recenter_coordinates(df) for df in all_segment_dfs]
    rotated_dfs = [rotate_to_initial_heading(df) for df in centered_dfs]

    #standardise per segment to get higher values and standard deviation
    for df in rotated_dfs:
        df["X_norm"] = (df["X_rot"] - df["X_rot"].mean()) / (df["X_rot"].std() + 1e-6)
        df["Y_norm"] = (df["Y_rot"] - df["Y_rot"].mean()) / (df["Y_rot"].std() + 1e-6)

    return rotated_dfs

def build_unique_windows(X_segments, y_segments, worm_ids, window=300):
    """
    Split each segment into non-overlapping windows.

    Parameters
    ----------
    X_segments : list[np.ndarray]
        Segment time series.
    y_segments : list[int]
        Segment labels.
    worm_ids : list[int]
        Worm identifier for each segment.
    window : int
        Window length (frames).

    Returns
    -------
    Xw : np.ndarray
        Windowed inputs.
    yw : np.ndarray
        Window labels.
    ww : np.ndarray
        Worm IDs per window.
    """

    Xw = []
    yw = []
    ww = []

    for seg, y, w in zip(X_segments, y_segments, worm_ids):
        n = seg.shape[0]
        for start in range(0, n, window):
            end = start + window
            if end <= n:
                Xw.append(seg[start:end])
                yw.append(y)
                ww.append(w)

    return np.array(Xw), np.array(yw), np.array(ww)

def build_overlap_windows(X_segments, y_segments, worm_ids, window=300, stride=150):
    """
    Split segments into overlapping windows.

    Parameters
    ----------
    X_segments : list[np.ndarray]
        Segment time series.
    y_segments : list[int]
        Segment labels.
    worm_ids : list[int]
        Worm identifier for each segment.
    window : int
        Window length (frames).
    stride : int
        Step size between windows.

    Returns
    -------
    Xw : np.ndarray
        Windowed inputs.
    yw : np.ndarray
        Window labels.
    ww : np.ndarray
        Worm IDs per window.
    """
    Xw = []
    yw = []
    ww = []

    for seg, y, w in zip(X_segments, y_segments, worm_ids):
        n = seg.shape[0]

        # Slide overlapping windows through the segment
        for start in range(0, n - window + 1, stride):
            end = start + window
            win = seg[start:end]
            
            Xw.append(win)
            yw.append(y)
            ww.append(w)

    return np.array(Xw), np.array(yw), np.array(ww)

def build_cnn_encoder(seq_len=300, num_features=4, embed_dim=64):
    """
    Build a 1D CNN encoder that maps a trajectory window to a fixed embedding.

    Parameters
    ----------
    seq_len : int
        Length of input sequence (frames).
    num_features : int
        Number of features per frame.
    embed_dim : int
        Size of output embedding.

    Returns
    -------
    tf.keras.Model
        CNN encoder model.
    """
    inp = Input(shape=(seq_len, num_features))

    
    x = Conv1D(64, 9, padding='same', activation='relu')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x) 

 
    x = Conv1D(128, 7, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(128, 5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    # temporal aggregation
    x = GlobalAveragePooling1D()(x)

    # embedding layer
    emb = Dense(embed_dim, activation='relu')(x)

    return Model(inp, emb, name="cnn_encoder")

def plot_training_history(history):
    """
    Plot training and validation accuracy and loss.

    Parameters
    ----------
    history : tf.keras.callbacks.History
        Training history returned by model.fit().
    """
    plt.figure(figsize=(12,4))

    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("CNN Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1,2,2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("CNN Loss")
    plt.legend()

    plt.show()


# Worm-level aggregation functions

def softmax_vote(probs, T=0.5):
    """
    Confidence-weighted softmax voting.
    Higher-confidence windows are up-weighted.
    """
    probs = np.asarray(probs)
    conf = np.abs(probs - 0.5)
    w = np.exp(conf / T)
    w /= w.sum()
    return np.sum(w * probs)


def median_vote(probs):
    """
    Median of window probabilities.
    """
    return np.median(probs)


def mean_vote(probs):
    """
    Mean of window probabilities.
    
    """
    return np.mean(probs)

def aggregate(worm_ids, y_true_windows, preds, method="softmax", T=0.5):
    """
    Aggregate window-level predictions into one prediction per worm.

    Parameters
    ----------
    worm_ids : array-like
        Worm ID for each window.
    y_true_windows : array-like
        True label for each window.
    preds : array-like
        Predicted probability for each window.
    method : str
        Aggregation method: "softmax", "median", or "mean".
    T : float
        Temperature for softmax voting.

    Returns
    -------
    pd.DataFrame
        Worm-level predictions and labels.
    """

    df = pd.DataFrame({
        "worm": worm_ids,
        "y_true": y_true_windows,
        "y_pred": preds
    })

    worm_results = []
    for w, g in df.groupby("worm"):
        probs = g["y_pred"].values

        if method == "softmax":
            worm_prob = softmax_vote(probs, T=T)
        elif method == "median":
            worm_prob = median_vote(probs)
        elif method == "mean":
            worm_prob = mean_vote(probs)
        else:
            raise ValueError("Unknown voting method")

        worm_results.append({
            "worm": w,
            "y_true": g["y_true"].values[0],
            "y_pred": worm_prob
        })

    worm_df = pd.DataFrame(worm_results)
    worm_df["y_hat"] = (worm_df["y_pred"] > 0.5).astype(int)
    return worm_df

def report(df, name):
    """
    Print classification metrics for worm-level predictions.
    """

    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(df["y_true"], df["y_hat"]))
    print(classification_report(df["y_true"], df["y_hat"]))