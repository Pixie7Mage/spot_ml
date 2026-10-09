import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from src.data_cleaning import preprocess_data
from src.feature_engineering import engineer_features
from src.config import CITY_COORDINATES

train_df = pd.read_csv('train-test.csv')
clean_tr, equip_meds, mkt_means = preprocess_data(train_df, CITY_COORDINATES)
fe_tr = engineer_features(clean_tr)

fe_tr['month'] = pd.to_datetime(fe_tr['date']).dt.month
fe_tr['ratio'] = fe_tr['posted_rate'] / (fe_tr['distance'] * fe_tr['quote_signal'])
fe_tr['log_ratio'] = np.log(fe_tr['ratio'])

NUM_FEATS = [
    'distance', 'log_dist', 'inv_dist',
    'clean_weight', 'log_weight', 'ton_miles', 'weight_per_mile', 'is_heavy', 'is_missing_weight',
    'clean_mkt', 'log_mkt', 'is_missing_mkt',
    'quote_signal', 'base_quote', 'mkt_adj_quote', 'rate_per_ton_mile_signal',
    'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon',
    'haversine', 'circuity', 'lat_diff', 'lon_diff', 'bearing',
    'dayofweek', 'is_weekend', 'dow_sin', 'dow_cos'
]
ALL_FEATS = NUM_FEATS + ['equipment']

tr_mask = fe_tr['month'] <= 8
val_mask = (fe_tr['month'] >= 9) & (fe_tr['month'] <= 10)

X_tr = fe_tr.loc[tr_mask, ALL_FEATS].copy()
y_tr = fe_tr.loc[tr_mask, 'log_ratio'].values

X_val = fe_tr.loc[val_mask, ALL_FEATS].copy()
y_val = fe_tr.loc[val_mask, 'posted_rate'].values
base_val = fe_tr.loc[val_mask, 'base_quote'].values

X_tr_lgb = X_tr.copy()
X_val_lgb = X_val.copy()
X_tr_lgb['equipment'] = X_tr_lgb['equipment'].astype('category')
X_val_lgb['equipment'] = X_val_lgb['equipment'].astype('category')

# Test varying num_leaves and min_child_samples
for nl in [31, 45, 63]:
    for lr in [0.02, 0.03, 0.04]:
        m = lgb.LGBMRegressor(
            objective='regression_l1',
            n_estimators=1000,
            learning_rate=lr,
            num_leaves=nl,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
        m.fit(X_tr_lgb, y_tr)
        p = base_val * np.exp(m.predict(X_val_lgb))
        mae = mean_absolute_error(y_val, p)
        print(f"num_leaves={nl}, lr={lr} -> MAE: {mae:.2f}")

# CatBoost depth tuning
X_tr_cb = X_tr.copy()
X_val_cb = X_val.copy()
X_tr_cb['equipment'] = X_tr_cb['equipment'].astype(str)
X_val_cb['equipment'] = X_val_cb['equipment'].astype(str)

for depth in [5, 6, 7]:
    m_cb = CatBoostRegressor(
        loss_function='MAE',
        iterations=1000,
        learning_rate=0.03,
        depth=depth,
        cat_features=['equipment'],
        random_seed=42,
        verbose=0,
        thread_count=-1
    )
    m_cb.fit(X_tr_cb, y_tr)
    p_cb = base_val * np.exp(m_cb.predict(X_val_cb))
    mae_cb = mean_absolute_error(y_val, p_cb)
    print(f"CatBoost depth={depth} -> MAE: {mae_cb:.2f}")
