import pandas as pd
import numpy as np
import os
import re
import glob
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error
from collections import Counter

# ---------------- CONSTANTES / OPTIONS ----------------
SEGMENTS_DIR = "preprocessed_data/segments"
SEGMENT_LENGTH = 900 
FEATURE_COLS = ['X', 'Y', 'Speed', 'turning_angle'] 
KEEP_SEGMENTS_WITH_DEATH_INSIDE = True  # True = keep segments that contain the death (target set to 0), False = exclude them

print("--- Lancement script avec Mask + cible depuis la FIN du segment ---")
print(f"Segment length: {SEGMENT_LENGTH}, features: {FEATURE_COLS}, KEEP_SEGMENTS_WITH_DEATH_INSIDE={KEEP_SEGMENTS_WITH_DEATH_INSIDE}")
print("-" * 60)

# ---------------- Chargement lifespans (robuste) ----------------
DATA_DIR = os.path.join(os.getcwd(), "TERBINAFINE")
LIFESPAN_FILE = os.path.join(DATA_DIR, "lifespan_summary.csv")

try:
    lifespan_df = pd.read_csv(LIFESPAN_FILE)
except Exception as e:
    print(f"ERREUR FATALE: impossible de lire {LIFESPAN_FILE}: {e}")
    raise SystemExit(1)

# Normaliser noms de colonnes (strip) et chercher colonne filename & lifespan
lifespan_df.columns = lifespan_df.columns.str.strip()
colmap = {c.lower(): c for c in lifespan_df.columns}

filename_col = colmap.get('filename') or colmap.get('file_name') or colmap.get('files') or colmap.get('source_file') or colmap.get('file')
life_col = colmap.get('lifespaninframes') or colmap.get('lifespan') or colmap.get('lifespan_in_frames') or colmap.get('lifespanframes') or colmap.get('total_frames')

if filename_col is None or life_col is None:
    print("Colonnes attendues introuvables dans lifespan_summary.csv. Colonnes disponibles:")
    print(list(lifespan_df.columns))
    raise SystemExit(1)

# Construire Worm_ID normalisé (sans extension, sans leading slash)
lifespan_df['Worm_ID_raw'] = lifespan_df[filename_col].astype(str)
lifespan_df['Worm_ID'] = lifespan_df['Worm_ID_raw'].apply(lambda x: os.path.splitext(x.strip())[0].lstrip('/'))
lifespan_map = lifespan_df.set_index('Worm_ID')[life_col].to_dict()

print(f"Loaded {len(lifespan_df)} lifespan entries. Example keys: {list(lifespan_map.keys())[:5]}")
print("-" * 60)

# ---------------- Lecture des segments et construction dataset avec mask ----------------
segment_files = glob.glob(os.path.join(SEGMENTS_DIR, "*.csv"))
all_segments = []
rejection_counts = Counter({
    'No_Lifespan_Match': 0,
    'Missing_Features': 0,
    'Pattern_Mismatch': 0,
    'After_Death': 0,
    'Other': 0
})

# pattern pour essayer d'extraire date/piworm/idx (ton pattern original)
SEGMENT_ID_PATTERN = re.compile(r'coordinates_highestspeed_(\d{8})_(\d+)_(\d+)$')

for file in segment_files:
    try:
        df = pd.read_csv(file)
    except Exception:
        rejection_counts['Other'] += 1
        continue

    # vérifier colonne source_file
    if 'source_file' not in df.columns:
        rejection_counts['Other'] += 1
        continue

    source_file = str(df['source_file'].iloc[0]).strip()
    worm_id_full = os.path.splitext(source_file)[0].strip().lstrip('/')
    worm_id_temp = re.sub(r'(-fragment\d+\.\d+-preprocessed)?(_with_time_speed)?$', '', worm_id_full)

    match = SEGMENT_ID_PATTERN.search(worm_id_temp)
    if match:
        date_part = match.group(1)
        plate_part = match.group(2).zfill(2)
        worm_part = match.group(3)
        worm_id_candidate = f"{date_part}_piworm{plate_part}_{worm_part}"
    else:
        # fallback: utiliser la string nettoyée sans leading slash
        worm_id_candidate = worm_id_temp.lstrip('/')

    # obtenir lifespan (essayer avec et sans leading slash)
    total_lifespan = lifespan_map.get(worm_id_candidate, None)
    if total_lifespan is None:
        total_lifespan = lifespan_map.get('/' + worm_id_candidate, None)
    if total_lifespan is None:
        rejection_counts['No_Lifespan_Match'] += 1
        continue

    # vérifier features
    missing_features = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_features:
        rejection_counts['Missing_Features'] += 1
        continue

    # frames / longueur
    try:
        segment_start_frame = int(df['GlobalFrame'].iloc[0])
    except Exception:
        rejection_counts['Other'] += 1
        continue

    original_length = df.shape[0]
    if original_length == 0:
        rejection_counts['Other'] += 1
        continue

    # Exclure segments qui commencent APRÈS la mort (incohérent)
    if segment_start_frame >= int(total_lifespan):
        rejection_counts['After_Death'] += 1
        continue

    # construire features array et padding
    features = df[FEATURE_COLS].values.astype(np.float32)
    current_length = features.shape[0]

    if current_length > SEGMENT_LENGTH:
        features = features[:SEGMENT_LENGTH, :]
        used_length = SEGMENT_LENGTH
    else:
        padding_needed = SEGMENT_LENGTH - current_length
        features = np.pad(features, ((0, padding_needed), (0, 0)), mode='constant', constant_values=0.0)
        used_length = current_length

    # construire mask (1 pour frames réelles, 0 pour padding)
    mask = np.zeros((SEGMENT_LENGTH,), dtype=np.float32)
    mask[:used_length] = 1.0

    # cible : time-to-death MESURÉ depuis la FIN du segment observé
    last_frame_observed = segment_start_frame + used_length - 1
    time_to_death_from_end = int(total_lifespan - last_frame_observed)

    if time_to_death_from_end < 0:
        # le ver est mort avant la fin du segment observé
        if KEEP_SEGMENTS_WITH_DEATH_INSIDE:
            target = 0.0
        else:
            rejection_counts['After_Death'] += 1
            continue
    else:
        target = float(time_to_death_from_end)

    # ajouter dans dataset (on stocke features, mask comme canal séparé plus tard)
    all_segments.append({
        'features': features,        # shape (SEGMENT_LENGTH, n_features)
        'mask': mask,               # shape (SEGMENT_LENGTH,)
        'y': target,
        'worm_id': worm_id_candidate,
        'start_frame': segment_start_frame,
        'valid_length': used_length
    })

print("---- Résumé chargement des segments ----")
print(f"Segments fichiers trouvés: {len(segment_files)}")
print(f"Segments acceptés : {len(all_segments)}")
print("Rejection counts:", dict(rejection_counts))
if len(all_segments) == 0:
    print("ERREUR: Aucun segment utilisable après filtrage.")
    raise SystemExit(1)
print("-" * 60)

# ---------------- Construire X (avec mask), y et groups ----------------
# On empile le mask comme dernier channel
X_raw = np.array([s['features'] for s in all_segments])  # shape (N, SEGMENT_LENGTH, n_feat)
masks = np.array([s['mask'] for s in all_segments])      # shape (N, SEGMENT_LENGTH)
y = np.array([s['y'] for s in all_segments], dtype=np.float32)
worm_groups = np.array([s['worm_id'] for s in all_segments])
start_frames = np.array([s['start_frame'] for s in all_segments])
valid_lengths = np.array([s['valid_length'] for s in all_segments])

# Concat features + mask channel (mask reste non-scaled)
mask_channel = masks.reshape(masks.shape[0], masks.shape[1], 1)
X_with_mask = np.concatenate([X_raw, mask_channel], axis=2)  # shape (N, SEGMENT_LENGTH, n_feat+1)

print(f"Shapes: X_with_mask={X_with_mask.shape}, y={y.shape}")
print("y distribution (describe):")
print(pd.Series(y).describe())
print("-" * 60)

# ---------------- Split (GroupShuffleSplit sans leakage) ----------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, temp_idx = next(gss.split(X_with_mask, y, groups=worm_groups))

gss_val_test = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx, test_idx = next(gss_val_test.split(X_with_mask[temp_idx], y[temp_idx], groups=worm_groups[temp_idx]))

X_train = X_with_mask[train_idx]
X_val = X_with_mask[temp_idx][val_idx]
X_test = X_with_mask[temp_idx][test_idx]

y_train = y[train_idx]
y_val = y[temp_idx][val_idx]
y_test = y[temp_idx][test_idx]

worm_groups_test = worm_groups[temp_idx][test_idx]
start_frames_test = start_frames[temp_idx][test_idx]

print("Split shapes:")
print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_val: {y_val.shape}, y_test: {y_test.shape}")
print("-" * 60)

# ---------------- Nettoyage NaN/Inf et Normalisation (scaler) ----------------
# On ne scale QUE les features (pas le mask). Le mask est le dernier channel.
n_features = len(FEATURE_COLS)
assert X_train.shape[2] == n_features + 1

def clean_arr(arr):
    # remplace inf par nan puis nan par 0
    arr = arr.copy()
    arr[~np.isfinite(arr)] = np.nan
    nan_count = np.count_nonzero(np.isnan(arr))
    if nan_count > 0:
        arr[np.isnan(arr)] = 0.0
    return arr, nan_count

# Nettoyage
X_train, nan_train = clean_arr(X_train)
X_val, nan_val = clean_arr(X_val)
X_test, nan_test = clean_arr(X_test)
print(f"NaN/Inf remplacés: train={nan_train}, val={nan_val}, test={nan_test}")

# Appliquer StandardScaler sur les features seulement (on aplati par frame)
def fit_and_transform_scaler(X_train, X_val, X_test, n_feat):
    # Extraire features
    train_feat = X_train[:, :, :n_feat].reshape(-1, n_feat)
    val_feat = X_val[:, :, :n_feat].reshape(-1, n_feat)
    test_feat = X_test[:, :, :n_feat].reshape(-1, n_feat)

    scaler = StandardScaler()
    scaler.fit(train_feat)

    train_feat_t = scaler.transform(train_feat).reshape(X_train.shape[0], X_train.shape[1], n_feat)
    val_feat_t = scaler.transform(val_feat).reshape(X_val.shape[0], X_val.shape[1], n_feat)
    test_feat_t = scaler.transform(test_feat).reshape(X_test.shape[0], X_test.shape[1], n_feat)

    # concat with mask channel (unchanged)
    X_train_scaled = np.concatenate([train_feat_t, X_train[:, :, n_feat:n_feat+1]], axis=2)
    X_val_scaled = np.concatenate([val_feat_t, X_val[:, :, n_feat:n_feat+1]], axis=2)
    X_test_scaled = np.concatenate([test_feat_t, X_test[:, :, n_feat:n_feat+1]], axis=2)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

X_train, X_val, X_test, scaler = fit_and_transform_scaler(X_train, X_val, X_test, n_features)

print("Post-scaling shapes:", X_train.shape, X_val.shape, X_test.shape)
print(f"Scaled X_train stats: min={np.min(X_train):.4f}, max={np.max(X_train):.4f}, mean={np.mean(X_train):.4f}")
print("-" * 60)

# ---------------- Modèle CNN (adapter input channels) ----------------
input_shape = (SEGMENT_LENGTH, n_features + 1)  # +1 pour le mask

def build_cnn_model(input_shape):
    model = Sequential([
        Conv1D(filters=32, kernel_size=10, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=10, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(50, activation='relu'),
        Dense(1, activation='linear')
    ])
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

model = build_cnn_model(input_shape)
print(model.summary())

# ---------------- Entrainement ----------------
print("Début entraînement...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

# ---------------- Evaluation ----------------
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"MAE global (frames): {mae:.2f}")

# prédictions et diagnostic par life stage
y_pred = model.predict(X_test).flatten()

results_df = pd.DataFrame({
    'worm_id': worm_groups_test,
    'start_frame': start_frames_test,
    'y_true': y_test,
    'y_pred': y_pred
})

# mapper worm lifespans
worm_lifespans = {}
for w in np.unique(worm_groups_test):
    if w in lifespan_map:
        worm_lifespans[w] = int(lifespan_map[w])
    elif ('/' + w) in lifespan_map:
        worm_lifespans[w] = int(lifespan_map['/' + w])

def get_life_stage(worm_id, start_frame, worm_lifespans):
    total_frames = worm_lifespans.get(worm_id)
    if total_frames is None or total_frames == 0:
        return 'Unknown'
    life_elapsed_pct = (start_frame / total_frames) * 100
    if life_elapsed_pct < 25:
        return 'Early (0-25%)'
    elif life_elapsed_pct < 75:
        return 'Mid (25-75%)'
    else:
        return 'Late (75-100%)'

results_df['life_stage'] = results_df.apply(
    lambda row: get_life_stage(row['worm_id'], row['start_frame'], worm_lifespans),
    axis=1
)
results_df['absolute_error'] = np.abs(results_df['y_true'] - results_df['y_pred'])

stage_mae = results_df.groupby('life_stage')['absolute_error'].mean().sort_values(ascending=False)
print("\nMAE par life stage (frames):")
print(stage_mae)
print("-" * 60)

# ---------------- Conseils rapides après exécution ----------------
print("Conseils:")
print("- Vérifie la distribution de y (beaucoup de zéros ?).")
print("- Si variance des durées entre worms est grande, teste une cible normalisée: y_frac = y / total_lifespan.")
print("- Si tu gardes KEEP_SEGMENTS_WITH_DEATH_INSIDE=True, contrôle les percentiles pour voir l'influence des zéros.")
print("- Tu peux aussi ajouter le channel 'valid_length / SEGMENT_LENGTH' si tu veux un signal global plutôt que mask per-frame.")

