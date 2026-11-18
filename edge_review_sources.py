"""Optional review inputs; failures never modify or block core projections."""
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import monotonic

import numpy as np
import pandas as pd
import requests

from live_projections import name_key

MINUTES_URL = 'https://www.rotowire.com/basketball/projected-minutes.php'
REFERENCE_COLUMNS = ['PLAYER_ID','Team','Opponent','game_date','minutes','source','updated_at','independent']


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
    columns=['PLAYER_ID','GAME_ID','POTENTIAL_AST','PASSES_MADE']
    if rows.empty:return pd.DataFrame(columns=columns)
    rows['day']=pd.to_datetime(rows.GAME_DATE,utc=True).dt.tz_convert('America/New_York').dt.date
    tables=read_days(season,set(rows.day));output=[]
    for day,boxes in rows.groupby('day'):
        if day not in tables:continue
        boxes=boxes[~boxes.PLAYER_ID.duplicated(keep=False)].set_index('PLAYER_ID')
        pair=boxes.join(tables[day],how='inner',rsuffix='_nba')
        good=pair.GP.eq(1)&pair.AST.eq(pair.AST_nba)&(pair.MIN-pair.MIN_nba).abs().le(.5+1/60)
        output.append(pair.loc[good].reset_index()[columns])
    return pd.concat(output,ignore_index=True) if output else pd.DataFrame(columns=columns)
