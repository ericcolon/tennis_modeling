import pandas as pd
import numpy as np
from scipy import sparse
from toolz import interleave


def get_player_mapping(_df, two_sided):
    assert not (_df['p1_id'] == 0).any()
    assert not (_df['p2_id'] == 0).any()
    if two_sided:
        # Need players and negations
        all_players = (
            set(_df['p1_id'].tolist()) |
            set((-_df['p1_id']).tolist()) |
            set(_df['p2_id'].tolist()) |
            set((-_df['p2_id']).tolist())
        )
    else:
        # Don't need negations
        all_players = (
            set(_df['p1_id'].tolist()) |
            set(_df['p2_id'].tolist())
        )
    inv_player_mapping = dict(list(enumerate(all_players)))
    player_mapping = {v: k for k, v in inv_player_mapping.iteritems()}
    return player_mapping, inv_player_mapping


def get_y(_df, feat, two_sided):
    if two_sided:
        y = np.concatenate([
            _df['p1_%s' % feat].values,
            _df['p2_%s' % feat].values
        ])
    else:
        y = _df[feat].values
    return y


def _get_forward_X(_df, player_mapping, two_sided):
    # Gets X with player 1 as server for server/returner features.
    n_matches = _df.shape[0]

    # Forward order
    p1_data = np.ones(n_matches)
    p1_row = np.arange(n_matches)
    p1_col = _df['p1_id'].map(lambda x: player_mapping[x]).values

    p2_row = np.arange(n_matches)
    if two_sided:
        # TODO: Should we negate 2nd player so best returns are positive?
        p2_data = np.ones(n_matches)
        p2_col = _df['p2_id'].map(lambda x: player_mapping[-x]).values
    else:
        p2_data = np.ones(n_matches) * -1
        p2_col = _df['p2_id'].map(lambda x: player_mapping[x]).values

    forward_data = np.hstack([p1_data, p2_data])
    forward_rows = np.hstack([p1_row, p2_row])
    forward_cols = np.hstack([p1_col, p2_col])

    return sparse.csc_matrix(
        (forward_data, (forward_rows, forward_cols)),
        shape=(n_matches, len(player_mapping))
    )


def _get_backward_X(_df, player_mapping):
    # TODO: This code can be simplified and merged with _get_forward_X
    # Gets X with player 2 as server (hence "backward") for server/returner features
    n_matches = _df.shape[0]

    back_p1_data = np.ones(n_matches)
    back_p1_row = np.arange(n_matches)
    back_p1_col = _df['p1_id'].map(lambda x: player_mapping[-x]).values

    back_p2_data = np.ones(n_matches)
    back_p2_row = np.arange(n_matches)
    back_p2_col = _df['p2_id'].map(lambda x: player_mapping[x]).values

    backward_data = np.hstack([back_p1_data, back_p2_data])
    backward_rows = np.hstack([back_p1_row, back_p2_row])
    backward_cols = np.hstack([back_p1_col, back_p2_col])

    return sparse.csc_matrix(
        (backward_data, (backward_rows, backward_cols)),
        shape=(n_matches, len(player_mapping))
    )


def get_X(_df, player_mapping, two_sided):
    '''
    Gets X data for regression features.  If two-sided, then two rows per match,
    one with each player as "server." If not two-sided, then one row per match with
    p1 index positive and p2 index negative.
    '''
    forward_X = _get_forward_X(_df, player_mapping, two_sided)
    if two_sided:
        backward_X = _get_backward_X(_df, player_mapping)
        return sparse.csc_matrix(sparse.vstack([forward_X, backward_X]))
    else:
        return forward_X


def get_X_y(_df, feat, player_mapping=None, inv_player_mapping=None, weights=None, two_sided=True):
    if weights is None:
        weights = np.ones(_df.shape[0])
    _df['__weight__'] = weights

    rel_df = _df[
        _df['p1_id'].notnull() &
        _df['p2_id'].notnull()
    ]
    if two_sided:
        rel_df = rel_df[
            _df['p1_%s' % feat].notnull() &
            _df['p2_%s' % feat].notnull()
        ]
    else:
        rel_df = rel_df[
            _df[feat].notnull()
        ]

    if player_mapping is None:
        player_mapping, inv_player_mapping = get_player_mapping(rel_df, two_sided=two_sided)

    y = get_y(rel_df, feat, two_sided=two_sided)
    X = get_X(rel_df, player_mapping, two_sided=two_sided)

    if two_sided:
        weights = np.concatenate([
            rel_df['__weight__'].values,
            rel_df['__weight__'].values
        ])
    else:
        weights = rel_df['__weight__'].values
    return X, y, player_mapping, inv_player_mapping, weights
