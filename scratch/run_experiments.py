import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_cleaning import preprocess_data
from src.feature_engineering import engineer_features
from src.config import CITY_COORDINATES, RANDOM_SEED

def main():
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

    print("="*60)
    print("WINDOW 3 HOLDOUT EVALUATION (Sep-Oct):")
    print(f"Train size: {tr_mask.sum()}, Val size: {val_mask.sum()}")
    print(f"Baseline MAE: {mean_absolute_error(y_val, base_val):.2f}")

    # 1. LightGBM L1
    X_tr_lgb = X_tr.copy()
    X_val_lgb = X_val.copy()
    X_tr_lgb['equipment'] = X_tr_lgb['equipment'].astype('category')
    X_val_lgb['equipment'] = X_val_lgb['equipment'].astype('category')

    m_lgb = lgb.LGBMRegressor(
        objective='regression_l1',
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=45,
        min_child_samples=25,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    m_lgb.fit(X_tr_lgb, y_tr)
    p_lgb = base_val * np.exp(m_lgb.predict(X_val_lgb))
    mae_lgb = mean_absolute_error(y_val, p_lgb)
    print(f"LightGBM MAE: {mae_lgb:.2f}")

    # 2. CatBoost MAE
    X_tr_cb = X_tr.copy()
    X_val_cb = X_val.copy()
    X_tr_cb['equipment'] = X_tr_cb['equipment'].astype(str)
    X_val_cb['equipment'] = X_val_cb['equipment'].astype(str)

    m_cb = CatBoostRegressor(
        loss_function='MAE',
        iterations=800,
        learning_rate=0.035,
        depth=6,
        cat_features=['equipment'],
        random_seed=42,
        verbose=0,
        thread_count=-1
    )
    m_cb.fit(X_tr_cb, y_tr)
    p_cb = base_val * np.exp(m_cb.predict(X_val_cb))
    mae_cb = mean_absolute_error(y_val, p_cb)
    print(f"CatBoost MAE: {mae_cb:.2f}")

    # 3. XGBoost MAE
    X_tr_xgb = pd.get_dummies(X_tr, columns=['equipment'])
    X_val_xgb = pd.get_dummies(X_val, columns=['equipment'])
    for c in X_tr_xgb.columns:
        if c not in X_val_xgb.columns:
            X_val_xgb[c] = 0
    X_val_xgb = X_val_xgb[X_tr_xgb.columns]

    m_xgb = XGBRegressor(
        objective='reg:absoluteerror',
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1
    )
    m_xgb.fit(X_tr_xgb, y_tr)
    p_xgb = base_val * np.exp(m_xgb.predict(X_val_xgb))
    mae_xgb = mean_absolute_error(y_val, p_xgb)
    print(f"XGBoost MAE: {mae_xgb:.2f}")

    # 4. Structural Ridge Base + Residual Tree Model
    # Fit Ridge on physics features: inv_dist, log_weight, log_mkt, equipment
    X_tr_ridge = pd.get_dummies(X_tr[['inv_dist', 'log_weight', 'log_mkt', 'equipment']], columns=['equipment'], drop_first=True)
    X_val_ridge = pd.get_dummies(X_val[['inv_dist', 'log_weight', 'log_mkt', 'equipment']], columns=['equipment'], drop_first=True)
    for c in X_tr_ridge.columns:
        if c not in X_val_ridge.columns:
            X_val_ridge[c] = 0
    X_val_ridge = X_val_ridge[X_tr_ridge.columns]

    m_ridge = Ridge(alpha=10.0)
    m_ridge.fit(X_tr_ridge, y_tr)
    ridge_tr_pred = m_ridge.predict(X_tr_ridge)
    ridge_val_pred = m_ridge.predict(X_val_ridge)

    # Residuals
    res_tr = y_tr - ridge_tr_pred
    m_lgb_res = lgb.LGBMRegressor(
        objective='regression_l1',
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=45,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    m_lgb_res.fit(X_tr_lgb, res_tr)
    p_res = base_val * np.exp(ridge_val_pred + m_lgb_res.predict(X_val_lgb))
    mae_res = mean_absolute_error(y_val, p_res)
    print(f"Ridge Base + LGB Residual MAE: {mae_res:.2f}")

    # 5. Ensembles
    weights = [
        (0.55, 0.45, 0.0),
        (0.50, 0.50, 0.0),
        (0.40, 0.40, 0.20),
        (0.45, 0.35, 0.20),
        (0.35, 0.45, 0.20)
    ]
    for w_l, w_c, w_x in weights:
        p_ens = w_l * p_lgb + w_c * p_cb + w_x * p_xgb
        mae_ens = mean_absolute_error(y_val, p_ens)
        rmse_ens = np.sqrt(mean_squared_error(y_val, p_ens))
        mape_ens = mean_absolute_percentage_error(y_val, p_ens) * 100
        wape_ens = np.sum(np.abs(y_val - p_ens)) / np.sum(y_val) * 100
        r2_ens = r2_score(y_val, p_ens)
        print(f"Ensemble ({w_l:.2f} LGB + {w_c:.2f} CB + {w_x:.2f} XGB) -> MAE: {mae_ens:.2f} | RMSE: {rmse_ens:.2f} | MAPE: {mape_ens:.2f}% | WAPE: {wape_ens:.2f}% | R2: {r2_ens:.4f}")

    # 6. Group Held-Out City Check
    print("\n" + "="*60)
    print("SPATIAL GROUP-HELD-OUT CHECK (Simulating Unseen Cities):")
    # Hold out 20% random cities from train set
    unique_cities = np.unique(np.concatenate([train_df['pickup'].dropna().unique(), train_df['delivery'].dropna().unique()]))
    np.random.seed(42)
    held_out_cities = set(np.random.choice(unique_cities, size=int(0.2 * len(unique_cities)), replace=False))
    print(f"Held-out cities ({len(held_out_cities)}): {list(held_out_cities)[:5]}...")
    
    city_val_mask = train_df['pickup'].isin(held_out_cities) | train_df['delivery'].isin(held_out_cities)
    city_tr_mask = ~city_val_mask

    X_c_tr = fe_tr.loc[city_tr_mask, ALL_FEATS].copy()
    y_c_tr = fe_tr.loc[city_tr_mask, 'log_ratio'].values
    X_c_val = fe_tr.loc[city_val_mask, ALL_FEATS].copy()
    y_c_val = fe_tr.loc[city_val_mask, 'posted_rate'].values
    base_c_val = fe_tr.loc[city_val_mask, 'base_quote'].values

    X_c_tr_lgb = X_c_tr.copy()
    X_c_val_lgb = X_c_val.copy()
    X_c_tr_lgb['equipment'] = X_c_tr_lgb['equipment'].astype('category')
    X_c_val_lgb['equipment'] = X_c_val_lgb['equipment'].astype('category')

    m_c_lgb = lgb.LGBMRegressor(objective='regression_l1', n_estimators=600, learning_rate=0.03, num_leaves=45, random_state=42, verbose=-1, n_jobs=-1)
    m_c_lgb.fit(X_c_tr_lgb, y_c_tr)
    p_c_lgb = base_c_val * np.exp(m_c_lgb.predict(X_c_val_lgb))
    mae_c_lgb = mean_absolute_error(y_c_val, p_c_lgb)
    base_c_mae = mean_absolute_error(y_c_val, base_c_val)
    print(f"Group Held-out Cities -> Baseline MAE: {base_c_mae:.2f} | LGB MAE: {mae_c_lgb:.2f} ({(1 - mae_c_lgb/base_c_mae)*100:.1f}% reduction)")

if __name__ == '__main__':
    main()
