from infra.defs import DATA_DIR
from scipy.sparse import csc_matrix, hstack
import numpy as np
import pandas as pd
import os


def get_match_result_data():
    match_result_dir = os.path.join(DATA_DIR, 'match_results')
    df = pd.concat([
        pd.read_csv(os.path.join(match_result_dir, f)) for f in os.listdir(match_result_dir)
    ])

    df.rename(columns={x: x.lower() for x in df.columns}, inplace=True)
    df.dropna(subset=['winner', 'loser'], inplace=True)
    df['wrank'].replace('NR', np.nan, inplace=True)
    df['lrank'].replace('NR', np.nan, inplace=True)
    df['wrank'] = df['wrank'].astype(float)
    df['lrank'] = df['lrank'].astype(float)
    df['match_id'] = range(df.shape[0])
    df['date'] = pd.to_datetime(df['date'])
    df['winner'] = df['winner'].map(lambda x: x.strip())
    df['loser'] = df['loser'].map(lambda x: x.strip())
    df['__surface__'] = df['surface'].copy()
    df.loc[df['court'] == 'Indoor', '__surface__'] = 'Indoor'

    player_set = set(df['winner']) | set(df['loser'])
    inverse_player_mapping = dict(list(enumerate(player_set)))
    player_mapping = {v: k for k, v in inverse_player_mapping.iteritems()}
    df['winner_idx'] = df['winner'].map(lambda x: player_mapping[x])
    df['loser_idx'] = df['loser'].map(lambda x: player_mapping[x])
    return df, player_mapping, inverse_player_mapping


def _get_sparse_X(df, n_players):
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


def _randomize_result(df):
    df['__chooser__'] = np.random.binomial(1, 0.5, size=df.shape[0])
    df['p1_idx'] = df.apply(
        lambda row: row['winner_idx'] if row['__chooser__'] == 1 else row['loser_idx'],
        axis=1
    )
    df['p2_idx'] = df.apply(
        lambda row: row['winner_idx'] if row['__chooser__'] == 0 else row['loser_idx'],
        axis=1
    )
    df['p1'] = df.apply(
        lambda row: row['winner'] if row['__chooser__'] == 1 else row['loser'],
        axis=1
    )
    df['p2'] = df.apply(
        lambda row: row['winner'] if row['__chooser__'] == 0 else row['loser'],
        axis=1
    )
    df['p1_rank'] = df.apply(
        lambda row: row['wrank'] if row['__chooser__'] == 1 else row['lrank'],
        axis=1
    )
    df['p2_rank'] = df.apply(
        lambda row: row['wrank'] if row['__chooser__'] == 0 else row['lrank'],
        axis=1
    )
    df['p1_odds'] = df.apply(
        lambda row: row['maxw'] if row['__chooser__'] == 1 else row['maxl'],
        axis=1
    )
    df['p2_odds'] = df.apply(
        lambda row: row['maxw'] if row['__chooser__'] == 0 else row['maxl'],
        axis=1
    )
    df['y'] = df['__chooser__']


def get_X_y(df, player_mapping, include_ranks=False, nan_fill=500.):
    if 'p1_idx' not in df or 'p2_idx' not in df:
        # Shuffle winners and losers so y can have both 0's and 1's.
        _randomize_result(df)
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

def sipko_weights(cur_date, train_df, disc, flat_time=1.):
    # These are the time decay weights used in the paper.
    # min(disc ^ (# years elapsed), disc ^ (flat_time))
    max_weight = disc ** flat_time
    days_ago = (pd.to_datetime(cur_date) - train_df['date']).map(lambda x: x.days)
    years_ago = days_ago / 365.
    weights = disc ** years_ago
    return weights.clip(upper=max_weight)
