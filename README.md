# NBA player archetypes — 2026–27 preseason

This update uses the user's 2026-10-18 minutes workbook and the repository's saved
2025–26 feature profiles to produce **provisional 2026–27 assignments**. No observed
2026–27 statistics are claimed. The source season remains explicit in every profile.

## Approved manual assignments

The user approved assignments for all 26 rotation players missing profiles, with four
changes to the proposed offensive roles: Kingston Flemings → Secondary Playmaker (6),
Yaxel Lendeborg → High Usage Wing (5), Darryn Peterson → Secondary Playmaker (6), and
Cameron Carr → Spot-Up Role Player (4). All other proposed offensive and defensive
assignments were accepted. The exact 26 pairs are in
`data/preseason/approved_archetypes.csv` and stored as final manual overrides.

There are now **26 offensive and 90 defensive override rules** (the original 64
plus the approved 26). All 285 players above 10 projected minutes have both archetypes;
**zero rotation players remain unassigned**. The 26 approved players still lack
historical feature profiles: manual labels do not create statistical observations.
The original 64 defensive overrides and all archetype names are unchanged.

## Completed run

* Imported **541 unique player IDs / 30 teams** from `Minutes!A1:G542` in
  `Custom_Team_Minutes_2026-10-18.xlsx`, preserving stored precision, all zeros,
  names, positions, starter selections and source row numbers. Every team totals
  240 projected minutes; nothing was rescaled. Workbook IDs are used directly;
  team abbreviations map to stable NBA franchise IDs. Workbook team membership is
  the user's roster scenario and was not independently verified against NBA rosters.
* Matched **381 players** to archived 2025–26 profiles. Of **285 players above
  10 projected minutes**, **259 were reclustered** and **26 lack a saved profile and now have approved manual assignments**.
  `preseason_archetypes.csv` includes all 541 players, including missing profiles
  and players outside the model's strict >10-minute cutoff. 318 players have positive
  minutes; 223 have zero. No missing profile is filled with invented numbers.
* Preserved **all 64 defensive override rules** and all original archetype names.
  **56 original defensive overrides** apply to this workbook, including 11 players outside
  the model's minutes cutoff. With the new approvals, 82 defensive and 26 offensive
  overrides apply. The other eight original rules remain stored for future rosters.
* Initialized **all team games played to zero**, based on the user's statement that
  no games have been played. The workbook itself does not supply game counts.
* The configurable opening window remains **10 completed team games**, an explicit
  implementation default, not an inferred user preference. Before game count 10,
  user minutes control eligibility and projection defaults; at 10 the normal runner
  uses current observed minutes. Preseason prior mode rejects nonzero team counts
  and does not automatically become a live-season model.

## Read the results

* `preseason_archetypes.csv`: full roster, projected minutes, offensive/defensive
  cluster IDs, names and assignment source. A blank cluster means no assignment.
* `player_clusters_detailed.csv`, `def_player_clusters_detailed.csv`: full roster
  with historical feature columns and provisional target-season assignments.
* `rotation_players_needing_profiles.csv`: the 26 >10-minute players without a
  saved historical profile; their approved manual archetypes are included.
* `rotation_players_needing_archetypes.csv`: unresolved rotation assignments; currently
  headers only because all 26 have been assigned.
* `missing_historical_profiles.csv`: all 160 players without saved profiles,
  including zero-minute players. No player is silently dropped.
* `team_minutes_check.csv`: independent aggregation of workbook team totals.
* `override_verification.csv`: every stored offensive/defensive override and its applied result
  or absence from the workbook.
* `season_refresh.json`, `minutes_import.json`: season, mode, input and reference
  hashes, workbook provenance, default assumptions and label alignment maps.

`SEASON_ID=22026` is the assignment season. `SourceSeasonID=22025` identifies observed
feature provenance. `FeatureDataStatus=preseason_historical_prior` marks reused
profiles. Missing profiles have no source season. `AssignmentSource` distinguishes
`kmeans_aligned`, `manual_override`, `outside_minutes_cutoff` and
`missing_historical_profile`. Eligibility and profile availability are separate flags.
A manual override can assign a player without model eligibility or a profile.

## Limitations of “last year's stats”

The upstream files are **partial-season 2025–26 snapshots, not verified full-season
2025–26 totals**. Their shot-profile `games_played` field ranges from 0 to 56. The
upstream repository supplies no raw `player_data.csv` or complete season export.
An NBA Stats connectivity check timed out; no fresh features were downloaded.
This run uses the available snapshots as the authorized temporary prior. Players
without them remain pending unless a manual rule supplies the assignment.

No older-year fallback, college-to-NBA conversion, teammate substitution, or position
based statistical profile is silently introduced. All 26 previously pending rotation players now have user-approved manual assignments.
Their underlying historical features remain missing. Full-season 2025–26 features
can replace the snapshot when available.

The original 34 offensive and 13 defensive features are standardized for KMeans
(`random_state=42`, `n_init=10`). Cluster counts stay at their historical values.
Fresh centroids are matched one-to-one to historical centroids using minimum
squared distance in the reference feature scale. This preserves label continuity
heuristically; basketball review remains useful. Unnamed historical defensive IDs
remain visible as `Cluster N (unnamed)`; no new descriptive role is invented.
The original offensive `Uncategorized` name is preserved.

Precedence: KMeans → aligned historical IDs → manual player override → archetype name.
Observed counts remain divided by observed `MIN` for per-36 features. Projected
minutes do not replace that denominator or create fake observations.

## Run or refresh

```sh
python -m pip install -r requirements.txt
python import_minutes.py /path/to/Custom_Team_Minutes_2026-10-18.xlsx --output data/preseason
python preseason_bootstrap.py --minutes data/preseason/projected_minutes.csv --output . --opening-games 10
python -m unittest discover -s tests -v
streamlit run main.py
```

The preseason dashboard reads the local roster results without requesting NBA game
logs or inventing games. It shows team filters, missing-profile flags, minutes and
assignments. While the prior manifest is active and all counts are zero, it opens
this view. The normal historical matchup/EV dashboard remains dependent on actual
game logs. Update team counts when games finish; rerun using real current profiles
when switching to observed-season data.

For an ordinary current-season refresh:

```sh
python season_pipeline.py --input player_data.csv --minutes projected_minutes.csv \
  --team-games team_games.csv --opening-games 10 --output .
```

`player_data.csv` must contain one enriched per-game current-season row per player
with `PLAYER_ID,PLAYER_NAME,SEASON_ID,MIN,HEIGHT,WEIGHT`, observed box scores and shot
location/tracking/hustle/play-type fields from `clustering_features.json`. Height
uses `6-8` syntax. Current input must have only target-season rows; historical input
is accepted only in explicit `--prior-season-id 22025` mode, which preserves source
season metadata and requires zero current-season games.

`projected_minutes.csv` requires `PLAYER_ID,TEAM_ID,SEASON_ID,PROJECTED_MIN`; additional
source/name metadata is retained by the importer. Minutes are numeric, 0–48, with
one row per player. `team_games.csv` requires `TEAM_ID,SEASON_ID,GAMES_PLAYED`, one row
per team with a nonnegative integer count of completed current regular-season games,
including games a player missed. Game counts must reflect the run date, independent
of dashboard date filters. Input duplicates, missing projections, invalid minutes,
wrong seasons and nonfinite model features are rejected.

`NBA_SEASON` defaults to `2026-27`; `NBA_OPENING_GAMES` defaults to `10`.
The dashboard accepts `NBA_MINUTES_FILE` and `NBA_TEAM_GAMES_FILE` overrides for its
normal projection view. Set the same opening window for refresh and dashboard.

## Sources and validation

* User workbook: `Custom_Team_Minutes_2026-10-18.xlsx`, `Minutes!A1:G542`. A SHA-256
  hash and exact row provenance are recorded. The workbook was read without edits.
* Historical snapshots and original rules: https://github.com/chadk94/Player_Type_Clustering
  at upstream commit `f683b29`, cloned 2026-10-19. Immutable snapshots are retained
  in `references/`; the root detailed exports now contain preseason results.
* Season API helper fixes: roster/game-log defaults, result filtering and dashboard
  labels now use season settings. Shot charts honor each row's requested season;
  enrichment seasons derive from input season IDs. Live feature ingestion has not
  been validated and is not required for this provisional run.

Automated checks cover exact override precedence, zero/decimal minutes, workbook ID
mapping and duplicate rejection, opening-window boundaries, label permutations,
rejection of unlabelled historical inputs, explicit source-year preservation,
missing-profile handling and full-roster preseason generation. The original aliases
for FanDuel spellings remain in `main.py`. Deployments should track the repository main branch and use main.py as the Streamlit entry point.

The 10 automated tests pass. A clean Python 3.12 installation of requirements.txt
also passed Streamlit AppTest checks for full app startup, the four adjusted offensive
roles, workbook minutes, the 318-player positive-minutes view, the full 541-player
view, and team filtering. First-time graphics initialization required cache warmup.
Live NBA APIs and hosted deployment status were not tested.
