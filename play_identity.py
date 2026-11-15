"""Stable identities for saved player props at a specific sportsbook."""
from hashlib import sha256
import json


def prop_id(play):
    parts = [play[k] for k in ('day', 'event', 'book', 'player_id', 'stat', 'side')]
    parts.append(format(play['line'], '.12g'))
    return sha256(json.dumps(parts, separators=(',', ':')).encode()).hexdigest()
