"""Validated daily schedules with a bounded retry for transient feed failures."""
import pandas as pd
import requests

from live_projections import ESPN, fetch_json, parse_schedule


class ScheduleUnavailable(ValueError):
    """The schedule could not be verified; this is not an empty slate."""


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
            return validate_daily_schedule(payload, int(season[:4])+1, date)
        except (requests.RequestException, ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
            if attempt == 1:
                raise ScheduleUnavailable(
                    f'ESPN schedule unavailable after two attempts ({type(exc).__name__}).'
                ) from exc
