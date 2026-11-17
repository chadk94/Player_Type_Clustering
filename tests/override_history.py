"""Reconstruct the October rules for the historical preservation checks."""
import csv

from season_pipeline import ROOT, load_rules


def october_rules():
    rules = load_rules()
    with (ROOT / 'data/audit/provisional_roles_approved_2026-11-17.csv').open() as source:
        for row in csv.DictReader(source):
            side = row['Side'].upper()
            role = rules[side + '_CLUSTER_OVERRIDES'].pop(row['PLAYER_NAME'])
            if rules[side + '_CLUSTER_NAMES'][str(role)] != row['Applied']:
                raise ValueError(f"Approved assignment differs for {row['PLAYER_NAME']}")
    return rules
