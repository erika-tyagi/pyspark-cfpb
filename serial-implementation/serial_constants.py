import warnings
import pandas as pd 
import numpy as np

from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

# Input files 
# Also accessible from http://files.consumerfinance.gov/ccdb/complaints.csv.zip 
COMPLAINTS_CSV = 'data/complaints.csv'

# Output files 
FEATURE_IMPORTANCE_OUT = 'results/feature_importance.json'
MODEL_EVALUATION_OUT = 'results/model_evaluation.csv'
MODEL_EVALUATION_OUT_FULL = 'results/model_evaluation_full.csv'
PROCESSED_DATA_DIR = 'data/processed_data'

# Column names 
RESPONSE_COL = 'company_response_to_consumer'
NARRATIVE_COL = 'consumer_complaint_narrative'
DATE_COL = 'date_received'

# Date range for complaints 
START_DATE = '2013-01-01'
END_DATE = '2020-06-01'

# Valid labels for complaints 
VALID_LABELS = [
    'Closed with explanation', 
    'Closed with monetary relief', 
    'Closed with non-monetary relief', 
    'Untimely response'
]

# Categorical features, maximum number of values to one-hot encode 
CAT_COLUMNS = [
    ('product', None), 
    ('issue', None), 
    ('company', 1000), 
    ('state', None), 
    ('tags', None), 
]

# Continuous features  
CONTINUOUS_COLUMNS = [
    'word_count', 
]

# k in k-fold cross validation 
K = 5

# Fraction of majority class to (under) sample 
FRAC_MAJORITY = 0.2

# Cross validation evaluation metric 
CV_METRIC = f1_score
CV_AVERAGE = 'micro'

# Models 
MODELS = {
    # 'LR':  OneVsRestClassifier(LogisticRegression()), 
    # 'DT':  OneVsRestClassifier(DecisionTreeClassifier()), 
    # 'SVM': OneVsRestClassifier(SVC(probability=True)), 
    'GBT': OneVsRestClassifier(GradientBoostingClassifier()), 
    'NB':  OneVsRestClassifier(GaussianNB())
}

# Model parameters 
PARAMETERS = {
    # 'LR':  {'estimator__penalty': ['l2'], 
    #         'estimator__C': [0.1, 1.0, 10.0]}, 
    # 'DT':  {'estimator__criterion': ['gini', 'entropy'], 
    #        'estimator__max_depth': [10, 20]}, 
    # 'SVM': {'estimator__C': [0.1], 
    #         'estimator__kernel': ['linear']}, 
    'GBT': {'estimator__learning_rate': [0.01, 0.1, 0.5], 
            'estimator__max_depth': [10, 20]}, 
    'NB':  {}
}

