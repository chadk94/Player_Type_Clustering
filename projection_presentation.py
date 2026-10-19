"""Display helpers; leave numeric model inputs and sportsbook calculations intact."""
import math


def over_line_label(line, price):
    if not all(math.isfinite(value) for value in (line,price)):
        return '—'
    return f'{line:g} {price:+.0f}'


def matchup_rating(baseline, projected, stat, enabled=True):
    """Five-point opponent lift scale; invert turnovers so higher means favorable."""
    if not enabled or not all(math.isfinite(value) for value in (baseline,projected)) or baseline <= 0:
        return '—'
    lift = round((projected/baseline-1)*100,8)
    if stat == 'TOV':
        lift = -lift
    if lift >= 10:
        return '5 ⬆️⬆️'
    if lift > 3:
        return '4 ⬆️'
    if lift >= -3:
        return '3 ↔️'
    if lift > -10:
        return '2 ⬇️'
    return '1 ⬇️⬇️'
