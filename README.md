# Laboratory of the Physics of Biological Systems (LPBS) - *C. elegans* Movement Analysis

## Project Overview

This project implements a comprehensive machine learning jupyter notebook for analyzing movement patterns of *Caenorhabditis elegans* worms tracked on plates. The system can classify worms that have been administered drugs versus control worms and predict death proximity in a featureless way.

**For more details and results see [Report.md](Report.md).**

### 1. Environment Setup
```bash
git clone https://github.com/titinelef/c-elegans-terminator.git
cd c-elegans-terminator
conda create -n celegans python=3.10
conda activate celegans
pip install -r requirements.txt
```
### 2. Data Processing Pipeline
Download the dataset 'TERBINAFINE' in the repo folder and run preprocessing.ipynb, the preprocessed data data will appear in a new folder "preprocessed_data" to use for the rest of the project.

Toward the end of the project, a second independent dataset was provided.
This dataset originates from a different experimental setup and is not used for training. It is used exclusively as a test set to:
Evaluate model robustness.
This external dataset is processed and evaluated at the end of the notebook: drug_prediction.ipynb

### 3. Project Overview

##  Structure


```text
.
├── preprocessing.ipynb                  # Raw data cleaning and preprocessing
├── death_regression.ipynb               # Time-to-death regression models
├── near_death_classification.ipynb      # Near-death classification
├── drug_prediction.ipynb                # Drug class prediction
│
├── attempts/                            # Experimental and exploratory notebooks
│   ├── attempt_death_regression.ipynb   # Alternative regression architectures and tests
│   ├── death_min_features.ipynb         # Death prediction with minimal feature sets
│   ├── death_proximity_min_features.ipynb # Near-death detection using reduced features
│   ├── drug_class_attempts.ipynb        # Experimental drug classification models
│   └── new_data_preprocessing.ipynb     # preprocessing pipeline for the new dataset given 
│
├── src/                                 # Helper modules shared across notebooks
│   ├── death_classifier_helpers.py      # Utilities for near-death classification
│   ├── drug_class_helpers.py            # Utilities for drug class prediction
│   ├── helpers_death_regression.py      # Regression models and evaluation helpers
│   └── preproc_helpers.py               # Preprocessing and data handling functions
│
├── preprocessed_data/                   # Generated after running preprocessing.ipynb
│   ├── segments/                        # Segmented time-series windows
│   └── full/                            # Full-length trajectories
│
├── Report.md                            # Details and results 
├── requirements.txt                    
└── README.md                          

```

The project is divided into **three distinct tasks** Each notebook is assigned to a specific task. 

#### 1. Near death classification 
**Notebook:** `near_death_classification.ipynb`

**Goal** : Can we predict whether a given worm trajectory segment corresponds to a near-death state

#### 2. Time-to-Death Regression  
**Notebook:** `death_regression.ipynb`

**Goal** : Can we predict when a worm will die based on its current physiological state?

#### 3. Drug treatment classification
**Notebook:** `drug_classification.ipynb`

**Goal** : Can we classify treated vs control worms from movement dynamics ?
