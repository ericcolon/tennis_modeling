from infra.defs import DATA_DIR
from scipy.sparse import csc_matrix, hstack
import numpy as np
import pandas as pd
import os


def _get_sparse_X(df, n_players):
    # Creates sparse matrix with which player indices are active
    n = df.shape[0]
    p1_data = np.ones(n)
    p1_row = np.arange(n)
    p1_col = df['p1_idx'].values

    p2_data = -np.ones(n)
    p2_row = np.arange(n)
    p2_col = df['p2_idx'].values

    all_data = np.concatenate([p1_data, p2_data])
    all_row = np.concatenate([p1_row, p2_row])
    all_col = np.concatenate([p1_col, p2_col])

    X = csc_matrix((all_data, (all_row, all_col)), shape=(n, n_players))
    return X

def get_X_y(df, player_mapping, include_ranks=False, nan_fill=500.):
    n_players = len(player_mapping)
    X = _get_sparse_X(df, n_players)
    assert ((X != 0).sum(axis=1) == 2).all()
    if include_ranks:
        ranks = csc_matrix(df[['p1_rank', 'p2_rank']].fillna(nan_fill).values)
        X = hstack([X, ranks])
    return X, df['y'].values


def _get_weights(cur_time, train_df, halflife):
    if halflife is None:
        weights = np.ones(train_df.shape[0])
    else:
        # NOTE: Can make this much faster by preloading integer days
        days_ago = (pd.to_datetime(cur_time) - train_df['date']).map(lambda x: x.days)
        lamb = np.log(2) / halflife
        weights = np.exp(-lamb * days_ago.astype(float))
    return weights

def sipko_weights(cur_date, train_df, disc, date_col='date', flat_time=1.):
    # Needs to be sorted prior to using
    # These are the time decay weights used in the paper.
    # min(disc ^ (# years elapsed), disc ^ (flat_time))
    max_weight = disc ** flat_time
    days_ago = (pd.to_datetime(cur_date) - train_df[date_col]).map(lambda x: x.days)
    years_ago = days_ago / 365.
    weights = disc ** years_ago
    return weights.clip(upper=max_weight)
