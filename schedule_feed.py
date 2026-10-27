"""Validated daily schedules and a bounded last-known-good fallback."""
from threading import RLock

import pandas as pd
import requests

from live_projections import ESPN, fetch_json, parse_schedule


class ScheduleUnavailable(ValueError):
    """The schedule could not be verified; this is not an empty slate."""


class ScheduleMemory:
    """Keep verified schedules through short outages, independently of feed TTLs.

    Shared by Streamlit sessions; only public schedule data is retained. A cached
    response keeps its original verification time, even after repeated reads.
    """
    max_age = pd.Timedelta(minutes=30)

    def __init__(self):
        self._frames = {}
        self._lock = RLock()

    def remember(self, season, date, frame):
        checked = pd.Timestamp(frame.attrs.get('schedule_checked_at'))
        if pd.isna(checked) or checked.tzinfo is None:
            return
        if (frame.attrs.get('schedule_season') != season
                or frame.attrs.get('schedule_date') != date.isoformat()):
            return
        key = (season, date)
        with self._lock:
            previous = self._frames.get(key)
            if previous is None or checked >= pd.Timestamp(previous.attrs['schedule_checked_at']):
                self._frames[key] = frame.copy(deep=True)
            # Bound memory even when many users browse different dates.
            if len(self._frames) > 64:
                oldest = min(self._frames, key=lambda k: self._frames[k].attrs['schedule_checked_at'])
                del self._frames[oldest]

    def recover(self, season, date, now=None):
        now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
        with self._lock:
            frame = self._frames.get((season, date))
            if frame is None:
                return None
            age = now - pd.Timestamp(frame.attrs['schedule_checked_at'])
            if age < pd.Timedelta(0) or age > self.max_age:
                return None
            backup = frame.copy(deep=True)
            backup.attrs['schedule_is_backup'] = True
            return backup


def validate_daily_schedule(payload, season_year, date):
    if not isinstance(payload, dict) or not isinstance(payload.get('events'), list):
        raise ValueError('Schedule response is missing its event list')
    event_ids = set()
    for event in payload['events']:
        if (not isinstance(event, dict) or not isinstance(event.get('season'), dict)
                or event['season'].get('year') != season_year):
            raise ValueError('Schedule returned a different or missing season')
        start = pd.Timestamp(event['date'])
        if pd.isna(start) or start.tzinfo is None or start.tz_convert('America/New_York').date() != date:
            raise ValueError('Schedule returned a different or invalid game date')
        event_id = str(event['id'])
        if event_id in event_ids:
            raise ValueError('Schedule returned duplicate games')
        event_ids.add(event_id)
        if event.get('season', {}).get('type') != 2:
            continue
        # Do not silently turn a damaged regular-season event into "no games".
        one = parse_schedule({'events': [event]}, season_year)
        completed = event.get('status', {}).get('type', {}).get('completed')
        if not completed and len(one) != 2:
            raise ValueError('Schedule returned an incomplete matchup')
    frame = parse_schedule(payload, season_year)
    if frame.TEAM_ID.duplicated().any():
        raise ValueError('Schedule returned more than one game for a team')
    return frame


def load_daily_schedule(season, date, fetcher=None):
    """Try at most twice; a verified empty event list is a successful response.

    fetch_json already bounds connection/read waits. Only daily schedule requests
    are retried here, so player changes do not retry league-wide data downloads.
    """
    fetcher = fetcher or fetch_json
    url = f'{ESPN}/site/v2/sports/basketball/nba/scoreboard?dates={date:%Y%m%d}'
    for attempt in range(2):
        try:
            payload = fetcher(url)
            frame = validate_daily_schedule(payload, int(season[:4])+1, date)
            frame.attrs.update(schedule_checked_at=pd.Timestamp.now(tz='UTC').isoformat(),
                               schedule_season=season, schedule_date=date.isoformat())
            return frame
        except (requests.RequestException, ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
            if attempt == 1:
                raise ScheduleUnavailable(
                    f'ESPN schedule unavailable after two attempts ({type(exc).__name__}).'
                ) from exc
