from fetch.defs import ROUND_MAP
from infra.defs import DATA_DIR
import os
import numpy as np
import pandas as pd
import json


def set_player_indices(df, player_mapping):
    df['p1'] = df[['winner', 'loser']].min(axis=1)
    df['p2'] = df[['winner', 'loser']].max(axis=1)
    df['p1_idx'] = df['p1'].map(lambda x: player_mapping[x])
    df['p2_idx'] = df['p2'].map(lambda x: player_mapping[x])

    switch_mask = df['winner'] > df['loser']
    df['p1_rank'] = df['wrank']
    df.loc[switch_mask, 'p1_rank'] = df.loc[switch_mask, 'lrank']
    df['p2_rank'] = df['lrank']
    df.loc[switch_mask, 'p2_rank'] = df.loc[switch_mask, 'wrank']

    df['p1_odds'] = df['maxw']
    df.loc[switch_mask, 'p1_odds'] = df.loc[switch_mask, 'maxl']
    df['p2_odds'] = df['maxl']
    df.loc[switch_mask, 'p2_odds'] = df.loc[switch_mask, 'maxw']

    df['p1_b365'] = df['b365w']
    df.loc[switch_mask, 'p1_b365'] = df.loc[switch_mask, 'b365l']
    df['p2_b365'] = df['b365l']
    df.loc[switch_mask, 'p2_b365'] = df.loc[switch_mask, 'b365w']

    df['p1_games'] = df['wgames']
    df.loc[switch_mask, 'p1_games'] = df.loc[switch_mask, 'lgames']
    df['p2_games'] = df['lgames']
    df.loc[switch_mask, 'p2_games'] = df.loc[switch_mask, 'wgames']

    df['p1_name'] = df['winner']
    df.loc[switch_mask, 'p1_name'] = df.loc[switch_mask, 'loser']
    df['p2_name'] = df['loser']
    df.loc[switch_mask, 'p2_name'] = df.loc[switch_mask, 'winner']

    df.loc[switch_mask, 'p2_odds'] = df.loc[switch_mask, 'maxw']
    df['y'] = (df['winner'] == df['p1']).astype(int)


def process_set_results(df):
    w_set_cols = ['w%d' % s for s in range(1, 6)]
    l_set_cols = ['l%d' % s for s in range(1, 6)]
    set_cols = w_set_cols + l_set_cols
    for set_col in set_cols:
        df[set_col] = df[set_col].replace(' ', None, inplace=False).astype(float)
    df['wgames'] = df[w_set_cols].sum(axis=1)
    df['lgames'] = df[l_set_cols].sum(axis=1)
    df['total_games'] = df[set_cols].sum(axis=1)


def featurize_fatigue_for_player(df, player, num_days=3):
    '''
    Featurizes fatigue for a single player
    <num_days>: Number of days before current match that count towards "fatigue"
    '''
    print "Featurizing %s..." % player
    player_df = df[
        (df['loser'] == player) |
        (df['winner'] == player)
    ].copy()

    player_df['last_retired'] = (
        (player_df['loser'] == player) &
        (player_df['comment'] == 'Retired')
    ).astype(int).shift(1)
    player_df['last_retired'].fillna(0, inplace=True)
    player_df.index = pd.to_datetime(player_df['date'])

    player_df['__count__'] = 1
    player_df['fatigue_matches'] = player_df.rolling('%dd' % num_days)['__count__'].sum() - 1
    player_df['fatigue_matches'] = player_df['fatigue_matches'].clip(lower=0.)

    # Subtract current games because rolling sum includes them...
    player_df['fatigue_games'] = player_df.rolling('%dd' % num_days)['total_games'].sum() - player_df['total_games']

    player_df['player'] = player

    player_df['time_since_last_match'] = player_df['date'].diff().map(
        lambda x: np.nan if pd.isnull(x) else x.days
    ).astype(float)

    return player_df[[
        'player',
        'match_id',
        'last_retired',
        'fatigue_matches',
        'fatigue_games',
        'time_since_last_match'
    ]]


def add_fatigue_features(df, comeback_def=20):
    '''
    Adds fatigue features
    <comeback_def>: How many days it needs to be since last match to be considered a "comeback"
    '''
    all_players = sorted(set(df['winner']) | set(df['loser']))
    player_dfs = [
        featurize_fatigue_for_player(df, p) for p in all_players
    ]
    player_feats = pd.concat(player_dfs)
    assert player_feats.shape[0] == df.shape[0] * 2
    # Merge in player feats
    feats = [col for col in player_feats.columns if col not in ('match_id', 'player')]  # TODO: Clean this hacky line up
    df = pd.merge(
        df,
        player_feats,
        left_on=['match_id', 'p1'],
        right_on=['match_id', 'player']
    )
    df.rename(columns={f: 'p1_%s' % f for f in feats}, inplace=True)
    df = pd.merge(
        df,
        player_feats,
        left_on=['match_id', 'p2'],
        right_on=['match_id', 'player']
    )
    df.rename(columns={f: 'p2_%s' % f for f in feats}, inplace=True)

    # Get diffs...
    df['fatigue_diff'] = df['p1_fatigue_matches'] - df['p2_fatigue_matches']
    df['fatigue_games_diff'] = df['p1_fatigue_games'] - df['p2_fatigue_games']
    df['time_diff'] = df['p1_time_since_last_match'] - df['p2_time_since_last_match']
    df['time_diff'].fillna(0, inplace=True)
    df['comeback_diff'] = 0
    df.loc[df['time_diff'] > comeback_def, 'comeback_diff'] = 1
    df.loc[df['time_diff'] < -comeback_def, 'comeback_diff'] = -1
    df['retire_diff'] = df['p1_last_retired'] - df['p2_last_retired']
    return df


def get_and_save_match_result_data():
    match_result_dir = os.path.join(DATA_DIR, 'match_results')
    print "Reading initial data..."
    df = pd.concat([
        pd.read_csv(os.path.join(match_result_dir, f)) for f in os.listdir(match_result_dir)
    ])
    print "Type converting and such..."
    df.rename(columns={x: x.lower() for x in df.columns}, inplace=True)
    df.drop(df[df['comment'] == 'Walkover'].index, inplace=True)  # Don't train/test on walkovers
    # Weird case in lsets
    df.drop(df[df['lsets'] == '`1'].index, inplace=True)
    df['lsets'] = df['lsets'].astype(float)
    df['wsets'] = df['wsets'].astype(float)

    df.dropna(subset=['winner', 'loser'], inplace=True)
    df['round'] = df['round'].map(lambda x: ROUND_MAP[x])
    df['wrank'].replace('NR', np.nan, inplace=True)
    df['lrank'].replace('NR', np.nan, inplace=True)
    df['wrank'] = df['wrank'].astype(float)
    df['lrank'] = df['lrank'].astype(float)
    df['match_id'] = range(df.shape[0])
    df['date'] = pd.to_datetime(df['date'])
    df['winner'] = df['winner'].map(lambda x: x.strip())
    df['loser'] = df['loser'].map(lambda x: x.strip())
    df['__surface__'] = df['surface'].copy()
    process_set_results(df)
    # Drop rows where we don't have game totals...
    df.drop(df[df['total_games'].isnull()].index, inplace=True)
    # TODO: Split out carpet court!
    df.loc[df['court'] == 'Indoor', '__surface__'] = 'Indoor'

    print "Getting player indices..."
    player_set = set(df['winner']) | set(df['loser'])
    inverse_player_mapping = dict(list(enumerate(player_set)))
    player_mapping = {v: k for k, v in inverse_player_mapping.iteritems()}

    print "Setting p1 and p2 instead of winner and loser..."
    set_player_indices(df, player_mapping)
    df.sort_values(by=['date', 'tournament', 'round'], inplace=True)

    df = add_fatigue_features(df)

    target_dir = os.path.join(DATA_DIR, 'parsed_match_results')
    if not os.path.exists(
        target_dir
    ):
        os.mkdir(target_dir)
    df.to_csv(os.path.join(target_dir, 'joined.tsv'), sep='\t', index=False)
    json.dump(player_mapping, open(os.path.join(target_dir, 'player_mapping.json'), 'w'))
    json.dump(inverse_player_mapping, open(os.path.join(target_dir, 'inverse_player_mapping.json'), 'w'))


def read_joined():
    target_dir = os.path.join(DATA_DIR, 'parsed_match_results')
    df = pd.read_csv(
        os.path.join(target_dir, 'joined.tsv'),
        sep='\t'
    )
    df['date'] = pd.to_datetime(df['date'])
    player_mapping = json.load(open(os.path.join(target_dir, 'player_mapping.json')))
    inverse_player_mapping = {v: k for k, v in player_mapping.iteritems()}
    # inverse_player_mapping = json.load(open(os.path.join(target_dir, 'inverse_player_mapping.json')))
    return df, player_mapping, inverse_player_mapping


if __name__ == '__main__':
    get_and_save_match_result_data()
