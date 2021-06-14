import numpy as np
import pandas as pd 
import json 
import time
import os

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, make_scorer, classification_report 

from serial_constants import * 

pd.options.mode.chained_assignment = None


def load_data(verbose=True):
    '''
    Loads and pre-processes data from the CFPB complaints database. 
    Returns: pandas df
    '''

    # Load complaints data  
    df = pd.read_csv(COMPLAINTS_CSV)

   # Clean column names 
    new_names = [n.lower().replace(" ", "_").replace("?", "") for n in df.columns]
    df.columns = new_names
    
    # Drop missing values in critical columns 
    df.dropna(subset=[RESPONSE_COL], inplace=True)

    # Limit to complaints in specified date range 
    df = df[(df.date_received >= START_DATE) & (df.date_received < END_DATE)]

    # Limit to complaints in specified outcomes 
    df = df[df[RESPONSE_COL].isin(VALID_LABELS)]
    
    # Convert dates 
    df.date_received = pd.to_datetime(df.date_received)
    df.date_sent_to_company = pd.to_datetime(df.date_sent_to_company)

    # Create week variable 
    df['week'] = df[DATE_COL].dt.strftime('%Y-w%V')
        
    # Print summary information 
    if verbose: 
        print('Date range: {} to {}'.format(df[DATE_COL].min(), df[DATE_COL].max()))
        print('Number of complaints: {}'.format(df.shape[0]))
        print('\n')
        print('Distribution of company response: \n{}'.format(df[RESPONSE_COL].value_counts() / df.shape[0]))
        print('\n')
        print('Number of unique values in each column: \n{}'.format(df.nunique()))
        
    return df


def process_features(df, narrative=False): 
    '''
    Engineers categorical and narrative features. One-hot-encodes categorical features and 
    creates word and character count features from narrative column. 

    Input: pandas df
    Returns: pandas df 
    '''

    # Process categorical features 
    return_df = process_cat_features(df)

    # Process narrative - keep indices aligned 
    if narrative: 
        char_count, word_count = process_narrative(df)
        return_df = return_df.assign(char_count=char_count.values)
        return_df = return_df.assign(word_count=word_count.values)

    return return_df


def process_cat_features(df):
    '''
    One-hot-encodes categorical features.

    Input: pandas df 
    Returns: pandas df 
    '''

    # Intialize output features dataset 
    return_df = pd.DataFrame() 
    
    for col_name, max_cats in CAT_COLUMNS: 

        # Replace values with Missing (if originally NA) or Other (if < max_cats)
        if not max_cats: 
            max_cats = df[col_name].nunique()

        top_n = df[col_name].value_counts()[:max_cats].index.tolist()
        return_df[col_name] = np.where(df[col_name].isin(top_n), df[col_name], 
                                       np.where(df[col_name].isna(), 'Missing', 'Other')) 
        
        # One-hot encode features 
        return_df = pd.get_dummies(return_df, columns=[col_name])
        
    return return_df


def process_narrative(df): 
    '''
    Creates character and word count features from the the narrative text field.  

    Input: pandas df
    Returns: 
    - character count (pandas series), word count (pandas series) 
    '''

    char_count = df[NARRATIVE_COL].str.len()
    word_count = df[NARRATIVE_COL].apply(lambda x: len(x.split()))
    
    return char_count, word_count


def get_count(series, val): 
    return int(series[series == val].shape[0])

def split_resample(X, y, normalize=True, resample=True, verbose=True): 
    '''
    Creates train-test-splits, normalizes continuous features, resamples training 
    data using SMOTE. 

    Input: 
    - X: pandas df or numpy array (features)
    - y: pandas series or numpy array (label)
    - normalize: boolean indicating whether to normalize continuous features 
    - resample: boolean indicating whether to resample minority classes in training data 

    Returns: 
    - X_train, X_test, y_train, y_test 
    - Also writes X_train, X_test, y_train, y_test to local CSV files 
    '''

    start_time = time.time()

    # Split into training and testing sets 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Normalize continuous features based on training set 
    CONTINUOUS_COLUMNS = []
    if normalize & len(CONTINUOUS_COLUMNS) > 0: 
        scaler = StandardScaler()
        X_train[CONTINUOUS_COLUMNS] = scaler.fit_transform(X_train[CONTINUOUS_COLUMNS])
        X_test[CONTINUOUS_COLUMNS] = scaler.transform(X_test[CONTINUOUS_COLUMNS])

    # Resample training data to balance classes 
    if resample: 
        # Under-sample majority classes 
        under_sampling_dict = {
            'Closed with explanation': int(get_count(y_train, 'Closed with explanation') * FRAC_MAJORITY), 
            'Untimely response': get_count(y_train, 'Untimely response'), 
            'Closed with non-monetary relief': get_count(y_train, 'Closed with non-monetary relief'), 
            'Closed with monetary relief': get_count(y_train, 'Closed with monetary relief')
        }

        rus = RandomUnderSampler(sampling_strategy=under_sampling_dict, random_state=0)
        X_train, y_train = rus.fit_resample(X_train, y_train)

        majority_samples = y_train[y_train == 'Closed with explanation'].shape[0] 

        # Over-sample minority classes 
        sm = SMOTE(random_state=0) 
        X_train, y_train = sm.fit_sample(X_train, y_train)

    # Write to csv 
    X_train.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train.csv'))
    y_train.to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv'))
    X_test.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test.csv'))
    y_test.to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_test.csv'))


    time_elapsed = time.time() - start_time 

    if verbose: 
        print('Distribution of training labels: \n{}'.format(y_train.value_counts() / y_train.shape[0]))
        print('\n')
        print('Training samples:', X_train.shape[0])
        print('Testing samples:', X_test.shape[0])
        print('Time elapsed:', time_elapsed)

    return X_train, X_test, y_train, y_test


def load_existing_split(verbose=True):
    '''
    Loads X_train, X_test, y_train, y_test (assuming split_resamples has already been run). 

    Returns: 
    - X_train, X_test, y_train, y_test 
    '''

    # Load existing csv files 
    X_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train.csv'), index_col=0)
    y_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv'), index_col=0)
    X_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test.csv'), index_col=0)
    y_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_test.csv'), index_col=0)

    # Convert y dataframes to series 
    y_train = y_train.iloc[:, 0]
    y_test = y_test.iloc[:, 0]

    if verbose: 
        print('Distribution of training labels: \n{}'.format(y_train.value_counts() / y_train.shape[0]))
        print('\n')
        print('Training samples:', X_train.shape[0])
        print('Testing samples:', X_test.shape[0])

    return X_train, X_test, y_train, y_test 


def find_best_model(X_train, y_train, verbose=True):
    '''
    Uses a grid search to identify the best classifier. Note that all grid search parameters 
    are specified in the constants file. 

    Input: 
    - X_train, y_train 

    Returns: 
    - best classifier (scikit-learn classifier) fit on the full X_train 
    - Also writes cross validation results to a local output file 
    '''
    
    # Specify cross-validation evaluation criteria 
    scorer = make_scorer(CV_METRIC, average=CV_AVERAGE)

    # Initialize best results dataframe 
    best_results_df = pd.DataFrame(columns=['classifier', 
                                        'mean_test_score', 
                                        'time', 
                                        'params'])

    # Initialize full results dataframe 
    all_results_df = []

    best_score = 0 
    best_clf = ''
    best_params = ''

    # Loop over models and parameters 
    for model_key in MODELS: 

        model = MODELS[model_key]
        param_values = PARAMETERS[model_key]

        start_time = time.time()

        print('Classifier:', model.estimator.__class__.__name__)

        # Run grid search 
        grid = GridSearchCV(estimator=model, 
                            cv=K, 
                            param_grid=param_values, 
                            scoring=scorer)

        grid.fit(X_train, y_train)

        time_elapsed = time.time() - start_time 

        # Store best models for each classifier method 
        best_results_df.loc[len(best_results_df)] = [model.estimator.__class__.__name__, 
                                                     grid.best_score_,
                                                     time_elapsed, 
                                                     grid.best_params_]

        # Store full results for each classifier method 
        clf_results_df = pd.DataFrame(grid.cv_results_)[['params', 'mean_test_score', 'mean_fit_time']]
        clf_results_df['classifier'] = model.estimator.__class__.__name__
        all_results_df.append(clf_results_df)  

        if verbose: 
            print('Best score:', grid.best_score_)
            print('Best parameters:', grid.best_params_)
            print('Time elapsed:', time_elapsed)
            print('\n')

        # Update overall best classifier 
        if grid.best_score_ > best_score: 
            best_score = grid.best_score_
            best_clf = grid.best_estimator_
            best_params = grid.best_params_


    # Write out model cross validation results 
    best_results_df.to_csv(MODEL_EVALUATION_OUT, index=False)

    all_results_concat = pd.concat(all_results_df)[['classifier', 'mean_test_score', 'mean_fit_time', 'params']]
    all_results_concat.to_csv(MODEL_EVALUATION_OUT_FULL, index=False)

    # Return overall best classifier trained on full training set 
    best_clf.set_params(**best_params)
    best_clf.fit(X_train, y_train)

    if verbose: 
        print('Overall best classifier:', best_clf.estimator.__class__.__name__)
        print('Overall best score', best_score)
        print('Best parameters:', best_params)

    return best_clf


def summarize_predictions(clf, y_true, y_pred):
    '''
    Summarizes predictions (i.e. highest probability class). 

    Input: 
    - clf: scikit-learn classifier 
    - y_true: true labels for testing set (pandas series)
    - y_pred: predicted labels for testing set (pandas series) (output of .predict)

    Returns: 
    - summary table (scikit-learn's classification report)
    '''

    summary = classification_report(y_true, y_pred, zero_division=0)
    print(summary)  

    return summary 


def summarize_probabilities(clf, y_true, y_pred_proba): 
    '''
    Summarizes prediction probabilility scores. Shows the average predicted probabilities for each 
    class based on the actual class label. 

    Input: 
    - clf: scikit-learn classifier 
    - y_true: true labels for testing set (pandas series)
    - y_pred_proba: predicted class probabilities (output of .predict_proba) 

    Returns: 
    - summary table (pandas df)
    '''

    proba_matrix = pd.DataFrame(y_pred_proba)
    proba_matrix.columns = clf.classes_
    proba_matrix = proba_matrix.assign(y_true=y_true.values)
    summary = proba_matrix.groupby('y_true').agg(lambda x: np.mean(x)).round(3)
    print(summary)

    return summary 


def feature_importance(clf, X_train, verbose=True): 
    '''
    Summarizes feature importance for each of the label classes. 

    Input: 
    - clf: scikit-learn classifier 
    - X_train (for feature names)

    Returns: 
    - feat_imp_dict: dictionary storing feature importance for each class 
    - Also writes dictionary to a local json file 

    To load json dictionary as a pandas df (e.g. for Closed class): 

        with open(FEATURE_IMPORTANCE_OUT as f:
            data = json.load(f)
        closed_feat_imp = pd.DataFrame(data['Closed'])
    '''

    # Initialize importance dictionary 
    feat_imp_dict = {} 

    # Manage feature importance for SVM 
    if clf.estimator.__class__.__name__ == 'SVC': 
        print('Feature importance not available for classifier')
    return feat_imp_dict

    # Loop over label classes 
    for i, label in enumerate(clf.classes_):
        coefs = pd.DataFrame({'feature': X_train.columns.values, 
                              'importance': clf.estimators_[i].feature_importances_.ravel()})
        coefs = coefs.sort_values(by='importance', ascending=False)
        
        if verbose: 
            print('Class:', label)
            print(coefs[:5])
            print('\n')
        
        feat_imp_dict[label] = coefs.to_dict()

    # Write to json 
    with open(FEATURE_IMPORTANCE_OUT, 'w') as fp:
        json.dump(feat_imp_dict, fp)
        
    return feat_imp_dict



