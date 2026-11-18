"""Optional review inputs; failures never modify or block core projections."""
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import monotonic
from functools import lru_cache
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import requests

from live_projections import name_key

MINUTES_URL = 'https://www.rotowire.com/basketball/projected-minutes.php'
REFERENCE_COLUMNS = ['PLAYER_ID','Team','Opponent','game_date','minutes','source','updated_at','independent']
PASSING_COLUMNS = ['PLAYER_ID','GAME_ID','POTENTIAL_AST','PASSES_MADE']
PASSING_ARCHIVE_COLUMNS = ['PLAYER_ID','GAME_ID','GAME_DATE','GP','MIN','AST','POTENTIAL_AST','PASSES_MADE']


@lru_cache(maxsize=2)
def archived_review_passing(season):
    """Public, game-level checkpoint for diagnostics only; rechecked against live boxes."""
    path = Path(__file__).parent / 'data' / f'review_passing_{season}.csv.gz'
    try:
        meta = json.loads(path.with_suffix('.source.json').read_text())
        if meta['season'] != season or hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256']:
            raise ValueError('Passing checkpoint scope or checksum mismatch')
        frame = pd.read_csv(path, dtype={'GAME_ID':str}, usecols=PASSING_ARCHIVE_COLUMNS)
        numeric = frame[['PLAYER_ID','GP','MIN','AST','POTENTIAL_AST','PASSES_MADE']]
        counts = frame[['PLAYER_ID','GP','AST','POTENTIAL_AST','PASSES_MADE']]
        if (len(frame) != meta['rows'] or frame.duplicated(['PLAYER_ID','GAME_ID']).any()
                or frame.GAME_DATE.isna().any() or frame.GAME_ID.isna().any()
                or not np.isfinite(numeric).all().all() or numeric.lt(0).any().any()
                or counts.mod(1).ne(0).any().any() or not frame.GP.eq(1).all()):
            raise ValueError('Invalid game-level passing checkpoint')
        return frame
    except (OSError, KeyError, TypeError, ValueError):
        return pd.DataFrame(columns=PASSING_ARCHIVE_COLUMNS)


class Node:
    def __init__(self, tag='', attrs=()):
        self.tag, self.attrs, self.children, self.text = tag, dict(attrs), [], ''

    def find(self, tag=None, cls=None):
        found=[]
        for child in self.children:
            if (tag is None or child.tag==tag) and (cls is None or cls in child.attrs.get('class','').split()):
                found.append(child)
            found.extend(child.find(tag,cls))
        return found

    def content(self):
        return self.text+''.join(c.content() for c in self.children)


class MinutesHTML(HTMLParser):
    def __init__(self):
        super().__init__(); self.root=Node(); self.stack=[self.root]

    def handle_starttag(self, tag, attrs):
        node=Node(tag,attrs); self.stack[-1].children.append(node)
        if tag not in {'img','br','hr','input','meta','link','source','area','base','embed','param','wbr'}:
            self.stack.append(node)

    def handle_endtag(self, tag):
        for i in range(len(self.stack)-1,0,-1):
            if self.stack[i].tag==tag:
                del self.stack[i:]; break

    def handle_data(self, data):
        self.stack[-1].text+=data


def parse_minutes(html, roster, games, retrieved_at):
    """Read visible public Today cards only, never subscriber placeholders.

    The page has no publisher update timestamp. updated_at records our retrieval,
    not a claim that the publisher revised its projection then.
    """
    stamp=pd.Timestamp(retrieved_at)
    if stamp.tzinfo is None:raise ValueError('Minutes retrieval must include a timezone.')
    day=stamp.tz_convert('America/New_York').date()
    opponents={}
    for game in games:
        tip=pd.Timestamp(game['GameTimeUTC'])
        if tip.tz_convert('America/New_York').date()==day:
            for team,opp in [(game['Away'],game['Home']),(game['Home'],game['Away'])]:
                if team in opponents:raise ValueError('Ambiguous minutes game.')
                opponents[team]=opp
    names={}
    for row in roster.itertuples():
        for alias in (row.PLAYER_NAME,getattr(row,'WorkbookPlayerName',None)):
            if pd.notna(alias):names.setdefault((row.TEAM_ABBREVIATION,name_key(alias)),set()).add(int(row.PLAYER_ID))
    parser=MinutesHTML();parser.feed(html);output=[]
    for card in parser.root.find(cls='proj-min'):
        if card.find(cls='lineups-viz-wall'):continue
        logos=card.find('img','lineup__logo')
        if len(logos)!=1:continue
        team=logos[0].attrs.get('alt')
        if team not in opponents:continue
        for entry in card.find('li','lineups-viz__player'):
            labels=entry.find(cls='lineups-viz__player-name'); meters=entry.find(cls='minutes-meter__range')
            if len(labels)!=1 or len(meters)!=1:continue
            ids=names.get((team,name_key(labels[0].content().strip())),set())
            if len(ids)!=1:continue
            try:minutes=float(meters[0].attrs['data-proj'])
            except (KeyError,ValueError):continue
            if not np.isfinite(minutes) or not 0<=minutes<=48:continue
            output.append(dict(PLAYER_ID=next(iter(ids)),Team=team,Opponent=opponents[team],game_date=day.isoformat(),
                minutes=minutes,source='RotoWire public game minutes',updated_at=stamp.isoformat(),independent=True))
    return pd.DataFrame(output,columns=REFERENCE_COLUMNS)


def fetch_minutes_page():
    response=requests.get(MINUTES_URL,timeout=(2,4))
    response.raise_for_status()
    return response.text,pd.Timestamp.now(tz='UTC').isoformat()


class MinutesReferenceFeed:
    """A slow optional source must not block changing games or minutes."""
    def __init__(self, fetch=fetch_minutes_page):
        self.fetch=fetch; self.pool=ThreadPoolExecutor(max_workers=1)
        self.lock=Lock(); self.pending=None; self.value=None; self.refresh_at=0

    def read(self):
        with self.lock:
            if self.pending is not None and self.pending.done():
                try:self.value=self.pending.result()
                except (requests.RequestException,ValueError,OSError):pass
                self.pending=None;self.refresh_at=monotonic()+300
            if self.pending is None and monotonic()>=self.refresh_at:
                self.pending=self.pool.submit(self.fetch)
            return self.value


def review_passing(season, logs, matches, read_days):
    """Verified daily opportunities, including games before the production seed.

    read_days can return immediately while its existing shared feed works in the
    background. Missing tracking leaves a review unchecked, never a zero count.
    """
    ids=pd.Series({pid:str(p.ESPN_PLAYER_ID) for pid,p in matches.items()},dtype=str)
    ids=ids[~ids.duplicated(keep=False)]
    rows=logs.copy()
    rows['PLAYER_ID']=rows.ESPN_PLAYER_ID.astype(str).map(dict(zip(ids,ids.index)))
    rows=rows[rows.PLAYER_ID.notna() & rows.MIN.gt(0)].copy()
    columns=PASSING_COLUMNS
    if rows.empty:return pd.DataFrame(columns=columns)
    rows['GAME_ID']=rows.GAME_ID.astype(str)
    rows['day']=pd.to_datetime(rows.GAME_DATE,utc=True).dt.tz_convert('America/New_York').dt.date
    rows=rows[~rows.duplicated(['PLAYER_ID','day'],keep=False)].copy()
    output=[]
    archive=archived_review_passing(season)
    if not archive.empty:
        pair=rows.merge(archive,on=['PLAYER_ID','GAME_ID'],how='inner',suffixes=('','_nba'),validate='one_to_one')
        good=(pair.day.astype(str).eq(pair.GAME_DATE_nba)&pair.GP.eq(1)&pair.AST.eq(pair.AST_nba)
              &(pair.MIN-pair.MIN_nba).abs().le(.5+1/60))
        output.append(pair.loc[good,columns])
    verified=set((pid,gid) for frame in output for pid,gid in zip(frame.PLAYER_ID,frame.GAME_ID))
    days=set(rows.loc[[(pid,gid) not in verified for pid,gid in zip(rows.PLAYER_ID,rows.GAME_ID)],'day'])
    # Never refetch the entire season on a cold app start just to review one game.
    # New or corrected boxes still require a fresh daily NBA match.
    tables=read_days(season,days) if days else {}
    for day,boxes in rows.groupby('day'):
        if day not in tables:continue
        boxes=boxes[~boxes.PLAYER_ID.duplicated(keep=False)].set_index('PLAYER_ID')
        pair=boxes.join(tables[day],how='inner',rsuffix='_nba')
        good=pair.GP.eq(1)&pair.AST.eq(pair.AST_nba)&(pair.MIN-pair.MIN_nba).abs().le(.5+1/60)
        output.append(pair.loc[good].reset_index()[columns])
    return (pd.concat(output,ignore_index=True).drop_duplicates(['PLAYER_ID','GAME_ID'],keep='last')
            if output else pd.DataFrame(columns=columns))
