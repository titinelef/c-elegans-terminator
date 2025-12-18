# Report

## Table of Contents
1. [Data Preprocessing](#1-data-preprocessing)  
2. [Near-Death Classification](#2-near-death-classification-near_death_classificationipynb)
3. [Time-to-Death Regression](#3-time-to-death-regression-death_regressionipynb)
4. [Drugged / Undrugged Classification](#4-drugged--undrugged-classification-drug_classificationipynb)

---
## 1. Data Preprocessing

### Phase 1: Raw Data → Preprocessed Data (`preprocessing.ipynb`)

- **Input:** Raw CSV Files (x, y, speed)
- **Processing Steps:**
  - Frame Reset handling
  - Turning angle calculation
  - Speed capping (removes outliers)
  - Gap cleaning (interpolate gaps with less than 7 consecutive missing values; remove large gaps)
  - Segment creation (split into 900-frame windows)
  - Exclude post-death segments
  - Coordinate normalization (scale to [0,1]) (used min and max values from `get_coordinate_bounds.py`)
  - Angle normalization (scale to [-1,1])
- **Output:** Preprocessed Data
  - `full/` folder: complete trajectories
  - `segments/` folder: 900-frame segments

### Preventing Data Leakage

A critical design constraint across all experiments is **worm-level isolation**:

- All splits (train / validation / test) are performed **at the worm level**
- No segments from the same worm appear in multiple splits
- Cross-validation uses **grouped CV (worm IDs as groups)**






---

## 2. Near-Death Classification (`near_death_classification.ipynb`)


#### Setup

- **Input:**
  - Speed  
  - Turning angle  

- **Label definition:**
  - Binary classification of whether a segment belongs to the last \(N\) segments before death  
  - \(N\) was first fixed to 20, then varied to assess sensitivity to the label definition

- **Models:**
  - Logistic Regression  
  - Random Forest  
  - Extra Trees  
  - HistGradientBoosting  
  - XGBoost  
  - SVM with RBF kernel and PCA  

- **Evaluation:**
  - Grouped cross-validation (worm-level)
  - Metrics: Accuracy, F1-score, AUC
  - Decision threshold tuned on a validation set

#### Results


- HistGradientBoosting achieves the strongest and most stable grouped cross-validation performance, while XGBoost reaches slightly higher peak performance after threshold tuning.
- Final results :

| Model           | Optimal threshold | Validation F1 |
|-----------------|-------------------|---------------|
| XGBoost         | 0.25              | 0.642         |
| SVM RBF + PCA   | 0.30              | 0.640         |
| ExtraTrees      | 0.45              | 0.640         |
| HistGB          | 0.45              | 0.638         |

- Performance depends strongly on the near-death definition:
  - For small \(N\), F1-scores are low (≈ 0.2–0.3)
  - Performance increases steadily as \(N\) increases
  - Best F1-scores reach **≈ 0.72–0.76** for \(N=30\)–40 segments

  


### Attempts (`death_min_features.ipynb` and `death_proximity_min_features.ipynb`)

In parallel to featureless time-series models, we explored near-death classification using a **minimal set of handcrafted aging features**.

- **Feature subsets tested:**
  - 1, 2, 3, 4, and 6 aging-related features
  - Full set of 29 aging features

- **Models evaluated:**
  - Gradient Boosting  
  - Logistic Regression  
  - Linear SVM (with calibration)  
  - XGBoost and LightGBM  

Using only a single feature (mean speed) already achieved reasonable performance (F1 ≈ 0.53, AUC ≈ 0.79). Expanding to 3 features (mean speed, speed entropy, mean roaming score) captured most of the predictive signal, reaching ~95% of the performance of the full feature set. More complex models and larger feature sets provided only marginal gains. 

---

## 3. Time-to-Death Regression (`death_regression.ipynb`)


#### Setup

- **Input:**
  - Speed  
  - Turning angle  

- **Target:**
  - Normalized remaining lifespan (fraction of life remaining in \([0,1]\))

- **Models:**
  - Mean regressor (baseline)
  - Bidirectional LSTM (Bi-LSTM)

- **Evaluation:**
  - Strict worm-level train / validation / test split
  - Metrics: Mean Absolute Error (MAE), \(R^2\)

#### Results

- The mean regressor shows no predictive power (\(R^2 \approx 0\)).
- The Bi-LSTM substantially outperforms the baseline:
  - MAE ≈ **0.16** (≈ 16% of total lifespan)
  - \(R^2 \approx 0.42\)
- These results indicate that short-term locomotor dynamics alone contain a strong signal of physiological age, despite biological variability and the limited temporal context of each segment.


### Attempts (`attempt_death_regression.ipynb`)

Several alternative regression approaches were explored in addition to the final Bi-LSTM model.

- **Baseline and classical models:**
  - Mean regressor (dummy baseline)
  - Linear regression models (Ridge, Lasso)
  - Tree-based regressors (Random Forest, Gradient Boosting)

- **Sequence-based models:**
  - Unidirectional LSTM
  - Deeper Bi-LSTM variants with different regularization strengths

Classical regressors consistently underperformed, with limited ability to capture temporal structure in the data. Tree-based models achieved moderate performance but were less stable across worms. Simpler LSTM architectures showed insufficient capacity. 

---

## 4. Drugged / Undrugged Classification (`drug_classification.ipynb`)



#### Additional Preprocessing for Drug Classification

Each 900-frame segment was represented using centered, rotated and normalized \(x,y\) coordinates, speed, and turning angle.  Segments were further split into overlapping 300-frame windows to increase sample count. 

#### Setup

- **Input:** :
  - Speed
  - Turning angle  
  - X_norm, Y_norm

- **Models:**
  - 1D CNN encoder
  - CNN embeddings followed by Gradient Boosting or CatBoost

- **Aggregation:** Softmax, mean, and median voting at worm level

#### Results

- Segment-level accuracy remains close to chance (≈ 0.5)
- Worm-level accuracy reaches at most **≈ 0.62**

###  Attempts (`drug_class_attempts.ipynb`)

Several deep learning approaches were explored for drug classification:
- CNN–LSTM hybrids on windows and full segments  
- Bidirectional LSTM models  
- Transformer-based temporal models on windowed inputs  

Across all models, accuracy was inconsistent and remained close to 0.5. 


### New Data Attempt (`new_data_preprocessing.ipynb` and `drug_classification.ipynb`)

When applied to a new dataset of 19 unseen worms, both CNN-based and feature-based models predicted **all worms as control**, with predicted probabilities clustered near the decision threshold.




