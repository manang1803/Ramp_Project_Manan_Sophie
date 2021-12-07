
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from skopt import BayesSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV

def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, 'year'] = X['date'].dt.year
    X.loc[:, 'month'] = X['date'].dt.month
    X.loc[:, 'day'] = X['date'].dt.day
    X.loc[:, 'weekday'] = X['date'].dt.weekday
    X.loc[:, 'hour'] = X['date'].dt.hour
    
    X.loc[:, 'hour_sin'] = np.sin(2*np.pi*X['hour']/24)
    X.loc[:, 'hour_cos'] = np.cos(2*np.pi*X['hour']/24)
    X.loc[:, 'month_sin'] = np.sin(2*np.pi*X['month']/12)
    X.loc[:, 'month_cos'] = np.cos(2*np.pi*X['month']/12)
    X.loc[:, 'weekday_sin'] = np.sin(2*np.pi*X['hour']/7)
    X.loc[:, 'weekday_cos'] = np.cos(2*np.pi*X['hour']/7)
    
    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date", 'month','weekday','hour']) 
    

def _merge_external_data(X):
    file_path = Path(__file__).parent / 'external_data.csv'
    df_ext = pd.read_csv(file_path, parse_dates=['date'], dtype={"flow_traffic": float, "traffic_occupation": float},low_memory=False)
    
    X = X.copy()
    # When using merge_asof left frame need to be sorted
    X['orig_index'] = np.arange(X.shape[0])
    X = pd.merge_asof(X.sort_values('date'), df_ext[['date','rr3','ff','t','vv','u','hbas','paris_covid','rafper','pm10','parks_change', 
    'residential_change', 'grocery_change', 'workplace_change','stations_change','retail_change','stringency_index','flow_traffic','traffic_occupation','vehicle_number']].sort_values('date'), on='date')
    X = X.sort_values('orig_index')
    del X['orig_index']
    return X

def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)
    date_cols = ['year', 'day', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos']
    categorical_cols = ["counter_name", "site_name","counter_id",'site_id']
    numerical_cols = ['t','rr3', 'ff','vv','hbas','rafper','paris_covid','pm10','parks_change', 'residential_change', 'grocery_change', 'workplace_change','stations_change','retail_change','flow_traffic','traffic_occupation','stringency_index','vehicle_number']
    
    preprocessor = ColumnTransformer([
        ('date', "passthrough", date_cols),
        ('cat', OneHotEncoder(handle_unknown = 'ignore'), categorical_cols),
        ('numeric', StandardScaler(), numerical_cols),
    ])

    pipe = make_pipeline(FunctionTransformer(_merge_external_data, validate=False),date_encoder, preprocessor,
             xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.8790730386941, gamma=0.41903774537530729, gpu_id=-1,
             importance_type='gain', interaction_constraints='',eval_metric='rmse',
             eta=0.05817619370698, max_depth=8, 
             min_child_weight=24.9870361960816, objective= 'reg:squarederror',
             n_estimators=705, n_jobs=8, num_parallel_tree=1, random_state=0,
             reg_alpha=6.3603597158578445, reg_lambda=2.3954299611600072, scale_pos_weight=1,
             subsample=0.6586297135971556, tree_method='exact', validate_parameters=1,
             verbosity=None))

    return pipe



