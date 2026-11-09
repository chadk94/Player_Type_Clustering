"""Display helpers; leave numeric model inputs and sportsbook calculations intact."""
import math


def over_line_label(line, price):
    if not all(math.isfinite(value) for value in (line,price)):
        return '—'
    return f'{line:g} {price:+.0f}'


def matchup_rating(baseline, projected, stat, enabled=True):
    """Five-point opponent lift scale; fewer turnovers/fouls mean favorable."""
    if not enabled or not all(math.isfinite(value) for value in (baseline,projected)) or baseline <= 0:
        return '—'
    lift = round((projected/baseline-1)*100,8)
    if stat in ['TOV','PF']:
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


def player_matchup_rating(player, stat, enabled=True):
    """Grade the archetype effect before team calibration. Threes use shot volume."""
    field = 'FG3A' if stat == '3PM' else stat
    baseline = player.get('Baseline ' + field, math.nan)
    if stat == 'PRA':
        baseline = sum(player.get('Baseline ' + part, math.nan) for part in ('PTS', 'REB', 'AST'))
    adjusted = player.get(field, math.nan)
    if stat == '3PM':
        adjusted = player.get('Pre-team 3PA', adjusted)
    elif stat in ('AST', 'REB'):
        adjusted = player.get('Pre-team ' + stat, adjusted)
    elif stat in ('PTS', 'PRA'):
        adjusted = player.get('Pre-market ' + stat, adjusted)
    return matchup_rating(baseline, adjusted, stat, enabled)
