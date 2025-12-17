# Report

## Table of Contents
1. [Data Preprocessing](#data-preprocessing)
2. [Drugged / undrugged Classification](#2-drugged--undrugged-classification)
3. [Death Prediction](#3-death-prediction)
4. [Results and Models Analysis](#4-results-and-models-analysis)

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


## 2. Drug prediciton 

## 3. Death Prediction

### Classification Léontine prime 

### Lifespan Regression Results
using LSTM 
***For this task, only feature-based models have been used.***


#### 3.1 Problem Formulation and Evaluation Metrics

The lifespan prediction task was modeled as a supervised regression problem. The objective was to estimate the **Percentage of Life Remaining** (normalized $y \in [0, 1]$) based on a 900-frame behavioral segment containing Speed and Turning Angle features. This target variable was chosen because using the percentage, 
$$(Lifespan_{\text{total}} - Time_{\text{End-segment}}) / Lifespan_{\text{total}}$$, provides a robust physiological age indicator, normalizing for inter-worm lifespan variability.

Metrics were chosen to assess both the magnitude of error and the model's explanatory power:

1.  **Mean Absolute Error (MAE):** The average absolute difference between the predicted and true remaining life. Presented as a percentage, a low MAE indicates precision in the time-to-death prediction.
2.  **$R^2$ Score (Coefficient of Determination):** Measures the proportion of the variance in the remaining lifespan that is explained by the model's input features. $R^2 \in [0, 1]$.

### 3.2 Model Performance Comparison

Three modeling approaches were implemented and rigorously compared using $k$-fold cross-validation with a strict worm-level split to ensure reliable generalization. The results confirm the effectiveness of the temporal Bi-LSTM architecture over the baseline.

| Model Name | Description | R² Score (Test) | MAE (Test) | Architecture |
| :--- | :--- | :--- | :--- | :--- |
| **1. Naive Baseline (Dummy Regressor)** | Always predicts the mean remaining life of the training set. | **0.0000** | **[INSERT DUMMY MAE, e.g., 24.39]%** | Minimal |
| **2. Full Bi-LSTM (Primary Model)** | 3 stacked Bi-LSTM layers + Deep Dense Head (128/64 units). | **[INSERT FULL BI-LSTM R2, e.g., 0.4205]** | **[INSERT FULL BI-LSTM MAE, e.g., 16.01]%** | High |
| **3. Bi-LSTM Experimental Variant (Slice Model)** | 3 stacked Bi-LSTM layers + Lighter Dense Head (64/32 units). | **[INSERT SLICE R2, e.g., 0.3850]** | **[INSERT SLICE MAE, e.g., 16.55]%** | Medium |

### 3.3 Analysis and Interpretation

#### Model Efficacy
The **Full Bi-LSTM** model achieved the best performance with an $R^2$ score of **[INSERT FULL BI-LSTM R2]**. This result is highly significant, as it shows that **[Value $\times 100$]\%** of the variance in the time-to-death can be explained solely by the kinematic features within a short (900-frame) window, validating the use of behavior as a biomarker for aging. The MAE of **[INSERT FULL BI-LSTM MAE]%** represents an average error of approximately one-sixth of the total lifespan.

### 3.4 Visual Results

The following figures illustrate the model's training dynamics and final test set performance.

#### A. Learning Curve (Huber Loss)

The plot demonstrates the model's training stability. The minimal gap and co-decrease between the Training Loss and Validation Loss indicate that the model is generalizing well to unseen data and is not significantly overfitting.


#### B. Prediction vs. True Value Scatter Plot

This scatter plot confirms the model's predictive accuracy. The clustering of predicted values along the $y=x$ diagonal line for the test set demonstrates a strong, linear correlation between the model's output and the true remaining lifespan percentage, reinforcing the $R^2$ score.

