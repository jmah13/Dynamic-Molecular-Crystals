import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import copy
import re
import statsmodels.formula.api as smf
from scipy import stats
import xgboost as xgb
from sklearn.model_selection import train_test_split
from eli5 import explain_weights


import statistical_analysis as sa
import plotting as pt

from scipy import stats


def split_datasets(X, y, test_size=0.5, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

### XG Boost realted
def xg_boost_get_data_and_labels_from_df(df, selected_col_types, label_name, features = None):
    xgboost_df = copy.copy(df[selected_col_types.index])

    print('One hot encoding of categorical columns:')
    for col_name in selected_col_types[selected_col_types['type'] == 'categorical'].index:
        dummy_columns = pd.get_dummies( xgboost_df[col_name])
        xgboost_df = pd.concat([xgboost_df, dummy_columns], sort=False)
        xgboost_df = xgboost_df.drop(columns=[col_name])
        print(f'\t{col_name} -> {dummy_columns.columns.values}')

    print('Re-scaling log columns:')
    for col_name in selected_col_types[selected_col_types['scale'] == 'log'].index:
        vals = xgboost_df[col_name].values
        prev_range = np.around([np.nanmin(vals), np.nanmax(vals)], decimals=3)
        new_vals = np.log(vals)
        new_range = np.around([np.nanmin(new_vals), np.nanmax(new_vals)], decimals=3)

        xgboost_df[col_name] = new_vals
        print(f'\t{col_name}: {prev_range} -> {new_range}')

    # drop entries for which label is nan
    xgboost_df = xgboost_df[xgboost_df[label_name].notna()]
    
    y = xgboost_df[label_name]

    if features is None:
        X = xgboost_df.drop(columns=[label_name])

    else: 
        X = xgboost_df[features]

    return X, y

def xg_boost_train(dtrain, dtest, params, num_boost_round = 999, early_stopping_rounds=10):
    evals_result = {}
    evals  = [(dtrain,'train'), (dtest, 'validation')]


    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=False
    )
    return model, evals_result

def xg_boost_simple_train_and_eval(df, selected_col_types, params, label_name):
    X, y = xg_boost_get_data_and_labels_from_df(df, selected_col_types, label_name)
    X_train, X_test, y_train, y_test = split_datasets(X, y, test_size=0.3, random_state=0)


    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    model, evals_result = xg_boost_train(dtrain, dtest, params, early_stopping_rounds=10)

    if selected_col_types.loc[label_name]['scale'] == 'log':
        title = f'{label_name} - log scale'
    else:
        title = f'{label_name}'

    pt.plot_training_metrics(evals_result, title=title)
    pt.plot_regression(model, dtest, y_test, title=title)
    display(explain_weights(model))

### Catboost related
from catboost import Pool, CatBoostClassifier, CatBoostRegressor

def catboost_get_data_and_labels_from_df(df, selected_col_types, label_name, features):
    catboost_df = copy.copy(df[selected_col_types.index])

    # print('Re-scaling log columns:')
    for col_name in selected_col_types[selected_col_types['scale'] == 'log'].index:
        vals = catboost_df[col_name].values
        prev_range = np.around([np.nanmin(vals), np.nanmax(vals)], decimals=3)
        new_vals = np.log(vals)
        new_range = np.around([np.nanmin(new_vals), np.nanmax(new_vals)], decimals=3)

        catboost_df[col_name] = new_vals
        # print(f'\t{col_name}: {prev_range} -> {new_range}')

    # drop entries for which label is nan
    catboost_df = catboost_df[catboost_df[label_name].notna()]

    y = catboost_df[label_name]

    if features is None:
        X = catboost_df.drop(columns=[label_name])

    else: 
        X = catboost_df[features]

    cat_features = []
    for i, col_name in enumerate(X.columns):
        if selected_col_types.loc[col_name, 'type'] == 'categorical':
            cat_features.append(i)
            X[col_name] = X[col_name].astype('str')

    # cat_features = np.where((selected_col_types['type'] == 'categorical').values)[0]
    return X, y, cat_features

def catboost_train_regressor(dtrain, deval, params):

    model = CatBoostRegressor(
        **params
    )

    model.fit(
        dtrain,
        eval_set=deval,
        verbose=False
    )

    evals_result = model.get_evals_result()

    return model, evals_result

def catboost_train_classifier(dtrain, deval, params):

    model = CatBoostClassifier(
        loss_function='MultiClass',
        **params
    )

    model.fit(
        dtrain,
        eval_set=deval,
        verbose=False,
    )

    evals_result = model.get_evals_result()

    return model, evals_result

def catboost_simple_train_and_eval(df, selected_col_types, params, label_name, features):
    X, y, cat_features = catboost_get_data_and_labels_from_df(df, selected_col_types, label_name, features)
    X_train, X_test, y_train, y_test = split_datasets(X, y, test_size=0.3, random_state=0)

    Q1 = np.percentile(y_train, 25)
    median = np.percentile(y_train, 50)
    Q2 = np.percentile(y_train, 75)

    Quartiles ={}
    Quartiles['Q1'] = Q1
    Quartiles['median'] = median
    Quartiles['Q2'] = Q2

    dtrain = Pool(data=X_train, label=y_train, cat_features=cat_features)
    deval = Pool(data=X_test, label=y_test, cat_features=cat_features)
    
    if selected_col_types.loc[label_name]['type'] == 'continous':
        model, evals_result = catboost_train_regressor(dtrain, deval, params)

    elif selected_col_types.loc[label_name]['type'] == 'categorical':
        model, evals_result = catboost_train_classifier(dtrain, deval, params)
    
    if selected_col_types.loc[label_name]['scale'] == 'log':
        title = f'{label_name} - log scale'
    else:
        title = f'{label_name}'

    pt.plot_training_metrics(evals_result, title=title)

    if selected_col_types.loc[label_name]['type'] == 'continous':
        pt.plot_regression(model, deval, y_test, y_train, Quartiles, title=title)
    elif selected_col_types.loc[label_name]['type'] == 'categorical':
        pt.plot_AUC(model, deval, y_test, title=title)

    display(explain_weights(model))


def get_augmented_data(df, equations, cols_to_augment):
    augmented_data = {}
    for i in cols_to_augment:
        for j in cols_to_augment:
            for equation in equations:
                data_name = f'{equation.__name__}({i}, {j})'
                vals = equation(df[i].values, df[j].values)
                augmented_data[data_name] = vals
                
    return pd.DataFrame(augmented_data)
        
