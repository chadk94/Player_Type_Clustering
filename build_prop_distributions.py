"""Select and validate count-prop models on chronological NBA box scores.

Usage: python build_prop_distributions.py PLAYER_GAMES.csv.gz ROLE_FORECAST_DIR
       ASSIST_FORECAST_DIR REPORT_DIR --profile data/prop_distribution_profile.json
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prop_distributions import COUNT_STATS, COUNT_VERSION, HIGH_SCALE, LOW_SCALE, LOW_WEIGHT, distribution_cdf


def forecast_samples(archive, role_dir, assist_dir):
    logs = pd.read_csv(archive, dtype={'GAME_ID': str})
    logs = logs.loc[logs.MIN.gt(0)].sort_values(['season', 'GAME_DATE', 'GAME_ID']).copy()
    if logs.duplicated(['season', 'PLAYER_ID', 'GAME_DATE']).any():
        raise ValueError('Ambiguous player/day')
    logs['3PM'] = logs.FG3M
    stats = ['PTS', *[s for s in COUNT_STATS if s != 'PRA']]
    columns = ['MIN', *stats]
    seasons = sorted(logs.season.unique())
    samples = []
    for old_season, season in zip(seasons, seasons[1:]):
        past = logs.loc[logs.season.eq(old_season)].groupby('PLAYER_ID')
        frame = logs.loc[logs.season.eq(season)].copy()
        current = frame.groupby('PLAYER_ID', sort=False)
        prev = past[columns].sum().reindex(frame.PLAYER_ID).set_axis(frame.index)
        previous_mpg = frame.PLAYER_ID.map(past.MIN.mean())
        minutes = current.MIN.transform(lambda s: s.shift().rolling(10, min_periods=1).mean()).fillna(previous_mpg).clip(upper=48)
        earlier = current[columns].cumsum() - frame[columns]
        taper = (20 - current.cumcount()).clip(0, 12) / 12
        means = pd.DataFrame(index=frame.index)
        for stat in stats:
            prior_minutes = np.where(prev.MIN.ge(500) & (stat in ('BLK', 'STL')), 1000, 200)
            weight = prior_minutes * taper * prev.MIN.notna()
            prior_rate = (prev[stat] / prev.MIN).fillna(0)
            means[stat] = minutes * (earlier[stat] + weight * prior_rate) / (earlier.MIN + weight)
        # Replace proxies with frozen pregame forecasts from the model studies wherever available.
        keys = ['GAME_ID', 'PLAYER_ID']
        means.index = pd.MultiIndex.from_frame(frame[keys])
        role = pd.read_csv(role_dir / f'predictions-{season}.csv.gz', dtype={'game': str})
        assert (role.last_training_day < role.day).all()
        role = role.rename(columns={'game':'GAME_ID', 'player':'PLAYER_ID'})
        role['stat'] = role.stat.replace({'FG3M':'3PM'})
        control = role.pivot(index=keys, columns='stat', values='control')
        means.update(control.drop(columns=['PRA', 'AST']))
        ast = pd.read_csv(assist_dir / f'forecasts-{season}.csv.gz', dtype={'GAME_ID': str})
        assert (ast.last_input_day.fillna('') < ast.GAME_DATE).all()
        assists = ast.set_index(keys).combined_200.rename('AST')
        means.update(assists.to_frame())
        means['PRA'] = means[['PTS','REB','AST']].sum(axis=1, min_count=3)
        frame['PRA'] = frame.PTS + frame.REB + frame.AST
        means.index = frame.index
        for stat in COUNT_STATS:
            part = frame[['season','GAME_DATE','GAME_ID','PLAYER_ID']].copy()
            part['stat'], part['actual'], part['mean'], part['minutes'] = stat, frame[stat], means[stat], minutes
            keep = part['mean'].ge(0) & np.isfinite(part['mean']) & part.minutes.gt(0)
            if stat == 'PF':
                keep &= part['mean'].le(6) & part.actual.le(6)
            samples.append(part.loc[keep])
    return pd.concat(samples, ignore_index=True)


def fit(frame, stat, method):
    if method in ('poisson', 'binomial6'):
        return dict(method=method, pool=0., players={})
    mu, y = frame['mean'], frame.actual
    variance = mu * (1 - mu / 6) if stat == 'PF' else mu
    exposure = 5 * variance if stat == 'PF' else mu**2
    values = pd.DataFrame({'extra': (y-mu)**2 - variance, 'exposure': exposure, 'count': 1})
    group = values.groupby(frame.PLAYER_ID).sum()
    pooled = values.extra.sum() / values.exposure.sum()
    def coefficient(raw):
        if method.endswith('_mixture'):
            second = LOW_WEIGHT * LOW_SCALE**2 + (1-LOW_WEIGHT) * HIGH_SCALE**2
            raw = (raw - second + 1) / second
        return np.clip(raw, 0 if stat == 'PF' else -.8, .8)
    players = {}
    if method.startswith('player'):
        # Forty prior appearances prevent noisy role-specific variance from dominating.
        raw = (group.extra / group.exposure).fillna(pooled)
        shrink = (group['count'] * raw + 40 * pooled) / (group['count'] + 40)
        players = {str(pid):round(value, 7) for pid,value in coefficient(shrink).items()}
    return dict(method=method, pool=round(float(coefficient(pooled)),7), players=players)


def scores(frame, stat, model, calibration=None):
    mu, actual = frame['mean'].to_numpy(), frame.actual.to_numpy()
    a = frame.PLAYER_ID.astype(str).map(model['players']).fillna(model['pool']).to_numpy()
    # The center and one count-scale standard deviation either side. Each appearance has equal weight.
    cuts = np.maximum(0, np.floor(mu[:,None] + np.sqrt(mu[:,None])*np.array([-1,0,1])))
    if stat == 'PF':
        cuts = np.minimum(cuts, 5)
    unique = np.column_stack([np.ones(len(frame),bool), np.diff(cuts,axis=1) != 0])
    p = 1 - distribution_cdf(cuts, mu[:,None], a[:,None], model['method'], stat)
    obs = actual[:,None] > cuts
    if calibration is not None:
        for left, right in zip([0,.1,.25,.4,.6,.75,.9], [.1,.25,.4,.6,.75,.9,1.]):
            in_bin = unique & (p >= left) & ((p < right) if right < 1 else (p <= 1))
            if in_bin.any():
                calibration.append(dict(stat=stat, low=left, high=right, lines=int(in_bin.sum()),
                                        predicted=float(p[in_bin].mean()), observed=float(obs[in_bin].mean())))
    brier = (((p - obs)**2 * unique).sum(axis=1) / unique.sum(axis=1)).mean()
    center = p[:,1]
    integer = np.maximum(0, np.floor(mu + .5))
    pmass = (distribution_cdf(integer,mu,a,model['method'],stat) -
             distribution_cdf(integer-1,mu,a,model['method'],stat))
    return dict(brier=float(brier), center_brier=float(np.mean((center-obs[:,1])**2)),
                center_pred=float(center.mean()), center_observed=float(obs[:,1].mean()),
                push_pred=float(pmass.mean()), push_observed=float((actual==integer).mean()),
                n=len(frame), games=int(frame.GAME_ID.nunique()))


def build(archive, role_dir, assist_dir, report_dir, profile_path):
    data = forecast_samples(archive, role_dir, assist_dir)
    metrics, summary, profiles, calibration = [], [], {}, []
    for stat in COUNT_STATS:
        sample = data.loc[data.stat.eq(stat)]
        dev = sample.loc[sample.season.eq('2024-25')]
        train = dev.loc[dev.GAME_DATE.le('2025-01-15')]
        validation = dev.loc[dev.GAME_DATE.gt('2025-01-15') & dev.minutes.ge(12)]
        holdout = sample.loc[sample.season.eq('2025-26')]
        test = holdout.loc[holdout.minutes.ge(12)]
        candidates = (['binomial6','pooled','player'] if stat == 'PF'
                      else ['poisson','pooled','player','pooled_mixture','player_mixture'])
        validation_scores = []
        for method in candidates:
            score = scores(validation, stat, fit(train,stat,method))
            metrics.append(dict(stat=stat,split='selection',method=method,**score))
            validation_scores.append(score['brier'])
        # Prefer the simpler candidate within 0.0001 absolute Brier of the minimum.
        winner = next(m for m,s in zip(candidates, validation_scores) if s<=min(validation_scores)+.0001)
        chosen = fit(dev, stat, winner)
        out = scores(test, stat, chosen, calibration)
        base = scores(test, stat, fit(dev,stat,'poisson'))
        metrics.extend([dict(stat=stat,split='test selected',method=winner,**out),
                        dict(stat=stat,split='test baseline',method='poisson',**base),
                        dict(stat=stat,split='test <12 minutes',method=winner,
                             **scores(holdout.loc[holdout.minutes.lt(12)],stat,chosen))])
        summary.append(dict(stat=stat,method=winner,baseline_brier=base['brier'],**out))
        profiles[stat] = fit(holdout,stat,winner)
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(report_dir/'validation.csv',index=False)
    pd.DataFrame(summary).to_csv(report_dir/'holdout.csv',index=False)
    pd.DataFrame(calibration).to_csv(report_dir/'calibration.csv',index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    profile = dict(model=COUNT_VERSION, training='2025-26', stats=profiles)
    profile['profile_hash'] = sha256(json.dumps(profile,sort_keys=True).encode()).hexdigest()
    profile_path.write_text(json.dumps(profile,sort_keys=True,allow_nan=False, separators=(',',':'))+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('archive','role_dir','assist_dir','report_dir','--profile'):
        parser.add_argument(arg,type=Path,**({'required': True} if arg.startswith('--') else {}))
    args = parser.parse_args()
    build(args.archive,args.role_dir,args.assist_dir,args.report_dir,args.profile)
