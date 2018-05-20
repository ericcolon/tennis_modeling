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
    df['y'] = (df['winner'] == df['p1'])


def get_and_save_match_result_data():
    match_result_dir = os.path.join(DATA_DIR, 'match_results')
    print "Reading initial data..."
    df = pd.concat([
        pd.read_csv(os.path.join(match_result_dir, f)) for f in os.listdir(match_result_dir)
    ])
    print "Type converting and such..."
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
    # TODO: Split out carpet court!
    df.loc[df['court'] == 'Indoor', '__surface__'] = 'Indoor'

    print "Getting player indices..."
    player_set = set(df['winner']) | set(df['loser'])
    inverse_player_mapping = dict(list(enumerate(player_set)))
    player_mapping = {v: k for k, v in inverse_player_mapping.iteritems()}

    print "Moving away from winner and loser..."
    set_player_indices(df, player_mapping)
    df.sort_values(by='date', inplace=True)
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
    inverse_player_mapping = json.load(open(os.path.join(target_dir, 'inverse_player_mapping.json')))
    return df, player_mapping, inverse_player_mapping


if __name__ == '__main__':
    get_and_save_match_result_data()
