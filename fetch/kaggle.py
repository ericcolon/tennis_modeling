from infra.defs import DATA_DIR
import pandas as pd
import os


def parse_scores(_df):
    # To write later
    pass


def _create_frac(_df, prefix, feat, denom):
    return (
        _df['%s_%s' % (prefix, feat)] /
        _df['%s_%s' % (prefix, denom)]
    )


def create_pct_features(_df):
    # Create percentage features
    for feat, denom, new_name in [
        ('ace', 'svpt', 'ace_pct'),
        ('1stIn', 'svpt', '1stIn_pct'),
        ('1stWon', '1stIn', '1stWon_pct'),
        ('2ndIn', '2ndpt', '2ndIn_pct'),
        ('2ndWon', '2ndIn', '2ndWon_pct'),
        ('svWon', 'svpt', 'svWon_pct')
    ]:
        for prefix in ('w', 'l'):
            _df['%s_%s' % (prefix, new_name)] = _create_frac(
                _df,
                prefix,
                feat,
                denom
            )


def preprocess(_df):
    # Creates new features that are differences of other features
    for prefix in ('w', 'l'):
        _df['%s_2ndpt' % prefix] = _df['%s_svpt' % prefix] - _df['%s_1stIn' % prefix]
        _df['%s_2ndIn' % prefix] = _df['%s_2ndpt' % prefix] - _df['%s_df' % prefix]
        _df['%s_svWon' % prefix] = _df['%s_1stWon' % prefix] + _df['%s_2ndWon' % prefix]


def parse_atp_data():
    target_path = os.path.join(DATA_DIR, 'ATP.csv')
    df = pd.read_csv(
        target_path,
        skiprows=[160637]
    )

    # Remove walkovers.  Don't want to train / test on those
    df.drop(
        df[
            df['score'].isnull() |
            (df['score'] == 'W/O')
        ].index,
        inplace=True
    )
    df['tourney_date'] = pd.to_datetime(
        df['tourney_date'],
        format='%Y%m%d'
    )
    parse_scores(df)
    preprocess(df)
    create_pct_features(df)

    df['p1_id'] = df[['winner_id', 'loser_id']].min(axis=1)
    df['p2_id'] = df[['winner_id', 'loser_id']].max(axis=1)
    switch_mask = df['winner_id'] > df['loser_id']
    for col in [
        'rank',
        'seed',
        'hand',
    ]:
        df['p1_%s' % col] = df['winner_%s' % col]
        df.loc[switch_mask, 'p1_%s' % col] = df.loc[switch_mask, 'loser_%s' % col]
        df['p2_%s' % col] = df['loser_%s' % col]
        df.loc[switch_mask, 'p2_%s' % col] = df.loc[switch_mask, 'winner_%s' % col]
    for col in [
        'ace_pct',
        '1stIn_pct',
        '1stWon_pct',
        '2ndIn_pct',
        '2ndWon_pct',
        'svWon_pct'
    ]:
        df['p1_%s' % col] = df['w_%s' % col]
        df.loc[switch_mask, 'p1_%s' % col] = df.loc[switch_mask, 'l_%s' % col]
        df['p2_%s' % col] = df['l_%s' % col]
        df.loc[switch_mask, 'p2_%s' % col] = df.loc[switch_mask, 'w_%s' % col]
    df['y'] = (df['winner_id'] == df['p1_id']).astype(int)
    df['match_id'] = range(df.shape[0])
    return df


def read_joined():
    # Incomplete, need parsers first!
    target_path = os.path.join(DATA_DIR, 'ATP.csv')
    df = pd.read_csv(
        target_path,
        skiprows=[160637]
    )
    df['tourney_date'] = pd.to_datetime(
        df['tourney_date'],
        format='%Y%m%d'
    )
    df['tourney_date'] = pd.to_datetime(
        df['tourney_date'],
        format='%Y%m%d'
    )
