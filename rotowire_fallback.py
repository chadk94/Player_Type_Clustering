"""Normalize an authorized RotoWire export into NBA-ID keyed per-game baseline rows."""
import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd

from live_projections import CORE_STATS, STAT_NAMES, name_key

COLUMN_NAMES = {'PLAYER':'PLAYER_NAME','NAME':'PLAYER_NAME','MPG':'MIN','MINUTES':'MIN',
                'G':'GP','3PM':'FG3M','3PTM':'FG3M','3PT':'FG3M','3PA':'FG3A','TO':'TOV',
                'OREBS':'OREB','DREBS':'DREB'}


def read_rotowire_csv(source):
    if isinstance(source,(str,Path)):
        text = Path(source).read_text(encoding='utf-8-sig')
    else:
        value = source.getvalue() if hasattr(source,'getvalue') else source.read()
        text = value.decode('utf-8-sig') if isinstance(value,bytes) else value.lstrip('\ufeff')
    for index,row in enumerate(csv.reader(io.StringIO(text))):
        keys = {str(c).strip().upper() for c in row}
        if keys.intersection({'NAME','PLAYER','PLAYER_NAME'}) and 'PTS' in keys and keys.intersection({'MIN','MPG','MINUTES'}):
            return pd.read_csv(io.StringIO(text),skiprows=index)
        if index >= 5:
            break
    raise ValueError('Could not find a RotoWire player-stat header in the first six rows')


def suffix_key(name):
    parts = name_key(name).split()
    if parts and parts[-1] in {'jr','sr','ii','iii','iv'}:
        parts.pop()
    return ' '.join(parts)


def normalize_rotowire(frame, roster, units='per_game'):
    if units not in {'per_game','season_totals','per_36'}:
        raise ValueError('Choose per_game, season_totals, or per_36 for RotoWire export units')
    frame = frame.copy()
    columns = {c:COLUMN_NAMES.get(str(c).strip().upper(),str(c).strip().upper()) for c in frame.columns}
    frame = frame.rename(columns=columns)
    if frame.columns.duplicated().any():
        raise ValueError('Ambiguous duplicate RotoWire columns')
    required = {'MIN', *CORE_STATS}
    if not required.issubset(frame.columns):
        raise ValueError('RotoWire export needs MIN/MPG, PTS, REB, AST, 3PM/FG3M, STL, BLK, TO/TOV')
    if 'PLAYER_ID' not in frame and 'PLAYER_NAME' not in frame:
        raise ValueError('RotoWire export needs Player names or explicit NBA PLAYER_ID values; RotoWire IDs are not NBA IDs')
    if 'PLAYER_ID' not in frame:
        identities, base_identities = {}, {}
        for row in roster.itertuples():
            for name in {row.PLAYER_NAME,getattr(row,'WorkbookPlayerName',row.PLAYER_NAME)}:
                identities.setdefault(name_key(name),set()).add(row.PLAYER_ID)
                base_identities.setdefault(suffix_key(name),set()).add(row.PLAYER_ID)
        def match(name):
            candidates = identities.get(name_key(name),set()) or base_identities.get(suffix_key(name),set())
            return next(iter(candidates)) if len(candidates)==1 else np.nan
        frame['PLAYER_ID'] = frame.PLAYER_NAME.map(match)
    else:
        frame['PLAYER_ID'] = pd.to_numeric(frame.PLAYER_ID,errors='coerce')
    unmatched = frame[~frame.PLAYER_ID.isin(roster.PLAYER_ID)].copy()
    frame = frame[frame.PLAYER_ID.isin(roster.PLAYER_ID)].copy()
    if frame.PLAYER_ID.duplicated().any():
        raise ValueError('Duplicate RotoWire rows for the same NBA player')
    for col in ['MIN',*STAT_NAMES]:
        frame[col] = pd.to_numeric(frame[col],errors='coerce') if col in frame else np.nan
    if units == 'season_totals':
        if 'GP' not in frame:
            raise ValueError('Season-total exports require GP to convert to per-game rates')
        gp = pd.to_numeric(frame.GP,errors='coerce')
        if not (gp.notna() & gp.gt(0) & np.isfinite(gp)).all():
            raise ValueError('Season-total exports require positive finite GP values')
        frame[['MIN',*STAT_NAMES]] = frame[['MIN',*STAT_NAMES]].div(gp,axis=0)
    elif units == 'per_36':
        # Normalize the rate denominator along with the stat numerators.
        frame['MIN'] = 36.
    core = frame[['MIN',*CORE_STATS]]
    if not np.isfinite(core.to_numpy()).all() or not frame.MIN.gt(0).all() or core.lt(0).any().any():
        raise ValueError('Matched RotoWire rows require finite, nonnegative core stats and a positive minutes denominator')
    if frame.empty:
        raise ValueError('No RotoWire player rows matched the workbook roster')
    return frame[['PLAYER_ID','MIN',*STAT_NAMES]], unmatched
