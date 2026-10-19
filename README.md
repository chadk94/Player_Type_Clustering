# NBA player archetypes — 2026–27 preseason

This update uses the user's 2026-10-18 minutes workbook and the repository's saved
2025–26 feature profiles to produce **provisional 2026–27 assignments**. No observed
2026–27 statistics are claimed. The source season remains explicit in every profile.

## Live player-stat projections

The deployed app now opens **Player vs Opponent**, with the original roster
available in a separate **Archetypes** tab. Live regular-season records and player
summaries refresh every five minutes while the page is open; a manual refresh button
clears the caches. Public ESPN endpoints provide the data because the official NBA
Stats/CDN requests tested from this environment were blocked or timed out.

The user selected this minutes policy: **workbook minutes for games 1–3 of each
team, then current-season average minutes from game 4 onward**. A team’s completed
regular-season wins plus losses drives the boundary, independently for each team.
The player’s season MPG divides minutes by appearances, excluding DNPs; the team
count is not the denominator. There is no post-window fallback to workbook minutes
for a player without matched season appearances. Unknown values remain blank.

Per-game stat estimates use full 2025–26 regular-season per-minute rates multiplied by
the selected minutes. These stay on the prior through team game 4. From game 5, current-season
per-minute rates receive weight `current minutes / (current minutes + 200)`; the
200-minute prior weight is an explicit smoothing assumption, separate from the
user’s three-game minutes rule. Players without a prior profile can receive estimates
from observed current-season stats from team game 5; a manual archetype alone cannot
create numeric rates. Counts, minutes sources, statistical baselines and missing-data
statuses are shown per player. Workbook zeroes are retained during the opening window.

The **Player vs Opponent** tab selects a player and any other team, with an opponent-adjustment toggle and baseline/adjusted stat comparison. The **Today’s Best Matchups** tab uses the day’s verified upcoming regular-season schedule (date selectable), ranks by any supported percentage, includes a negative-matchup toggle and minimum-sample filter, and displays a green/red heatmap, home/away, archetypes, sample counts, projected lines and CSV download. Empty slates stay empty; choose another date for previews. Rosters remain the user's workbook scenario, not confirmed lineups.

`matchup_projections.py` restores the legacy app's per-36 archetype-versus-opponent model. Only player-games with at least five minutes qualify. Each archetype's mean per-36 rate against an opponent is compared with its overall mean; deviations decay with half-lives of 15 samples for AST, 10 for OREB/DREB/STL/BLK and 999 for other stats, then shrink toward zero with three neutral samples. Offensive clusters drive PTS/AST/OREB/shooting/TOV; defensive clusters drive DREB/STL/BLK. Three-point makes use the attempts percentage to preserve individual conversion rates. REB = OREB + DREB and PRA = PTS + REB + AST. Sample counts are **player-games**, not unique team games. Missing/zero cluster baselines or missing matchup history remain blank, never an assumed neutral adjustment.

**Game-5 policy:** player stat-rate blending begins after the player's team completes four games; opponent tendencies use last season until the opponent completes four games, then use only current-season games. The minutes boundary remains game 4. This separates a player's rate sample from the opponent's defensive sample and lets teams cross the boundary independently. Current samples remain small, so the model retains neutral shrinkage and exposes counts. No pace, injury, starting-lineup or home-court adjustment is applied.

`data/game_logs_2026.csv.gz` contains 26,547 player appearances across all **1,230 2025–26 regular-season games**, retrieved 2026-10-19 from ESPN team schedules and summary box scores. Each of 30 teams has 82 games. The NBA Cup Championship is explicitly excluded even though ESPN tags it season type 2. Source details are in `data/game_logs_2026.csv.source.json`. Current approved archetypes/manual overrides label these historical appearances; archived labels cover players outside the current roster. Exact unambiguous canonical/workbook name matching links providers; unmatched players do not contribute cluster samples. Full-season averages derived from these box scores supersede partial archived stat baselines where matched; archived feature data remains unchanged for clustering. RotoWire still covers players without actual prior-season profiles.

Rebuild the historical snapshot with `python matchup_data.py --year 2026 --cache /path/to/local/boxscore-cache`. The cache holds raw source JSON outside the repository. Current-season schedules/box scores refresh every five minutes once a team reaches the matchup boundary; completed box scores are cached locally to avoid downloading them again. Every box score must match the requested season, type and completed status. Partial feed failures produce a visible warning.

Data guards require all 30 current-season regular-season team records. ESPN may
silently return last year’s player summaries before games exist; the app explicitly
rejects mismatched `requestedSeason` years and preseason/playoff summaries. Team
counts of zero do not trigger a request for preseason player stats. Cross-provider
player matches are exact after punctuation/diacritic normalization, including both
canonical and workbook spellings; ambiguous names remain unavailable. If traded
players have overlapping summaries, the largest available sample is selected rather
than adding records that might double-count games. If team records cannot be loaded,
projections pause rather than assuming zero games or continuing workbook minutes.

Current endpoints: ESPN `/apis/v2/sports/basketball/nba/standings`,
`/apis/site/v2/sports/basketball/nba/teams/{id}/athletes/statistics`, and
`/apis/site/v2/sports/basketball/nba/scoreboard`. ESPN uses the season-ending year
(2027 for 2026–27); NBA IDs and archetype assignments remain unchanged.

## Approved manual assignments

The user approved assignments for all 26 rotation players missing profiles, with four
changes to the proposed offensive roles: Kingston Flemings → Secondary Playmaker (6),
Yaxel Lendeborg → High Usage Wing (5), Darryn Peterson → Secondary Playmaker (6), and
Cameron Carr → Spot-Up Role Player (4). All other proposed offensive and defensive
assignments were accepted. The exact 26 pairs are in
`data/preseason/approved_archetypes.csv` and stored as final manual overrides.

There are now **28 offensive and 92 defensive override rules** (the original 64
defensive rules, the approved 26 player pairs, and two subsequent player corrections). All 285 players above 10 projected minutes have both archetypes;
**zero rotation players remain unassigned**. The 26 approved players still lack
historical feature profiles: manual labels do not create statistical observations.
The original 64 defensive overrides and all archetype names are unchanged.

### Knicks corrections (2026-10-19)

OG Anunoby and Josh Hart were both automatically reassigned to offensive cluster 3
(`Uncategorized`) and defensive cluster 3 (`Bad/Slow Big Defender`) by the preseason
KMeans rerun and historical-centroid alignment. Neither had an override. Their
archived labels were different (OG offense 4, Hart offense 6, both defense 8), so this
was a poor automatic assignment, not a stale UI label or a missing-stat fallback.

At the user's request, analyst corrections now override the model: **OG Anunoby →
Catch & Shoot Wing (0) / Big Wing Defender (5)**; **Josh Hart → Secondary Playmaker
(6) / Big Wing Defender (5)**. These are role-based judgments within the existing
archetype vocabulary: OG's off-ball scoring and versatile wing defense, and Hart's
connector/secondary-creation role and wing defense. Supporting role descriptions:
[NBA on Anunoby's cutting and wing role](https://www.nba.com/news/the-the-athletic-how-will-knicks-new-coach-mike-brown-deploy-consummate-cutter-og-anunoby),
[NBA on Hart's impact beyond scoring](https://www.nba.com/news/josh-hart-finals-impact).
The correction audit is `data/preseason/archetype_corrections.csv`; these are analyst
choices authorized by the correction request, distinct from the original 26 approved
pairs. Both overrides apply to roster exports, future clustering refreshes, historical
matchup sample labels and live projections. Regression tests deliberately feed the
wrong cluster-3 model assignments and verify override precedence and sample grouping.
Other players' assignments and all minutes are unchanged.

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
* The configurable opening window remains **3 completed team games**, an explicit
  implementation default, not an inferred user preference. Before game count 3,
  user minutes control eligibility and projection defaults; at 3 the normal runner
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
python preseason_bootstrap.py --minutes data/preseason/projected_minutes.csv --output . --opening-games 3
python -m unittest discover -s tests -v
streamlit run main.py
```

The live projection dashboard reads the approved roster and fetches validated current
regular-season counts/averages. The zero-game `team_games.csv` remains a reproducible
preseason bootstrap input; it is **not** used for the live projection minutes policy.
The historical matchup/EV code is retained as a fallback entry for other seasons,
but the current requested screen does not use odds or betting APIs.

For an ordinary current-season refresh:

```sh
python season_pipeline.py --input player_data.csv --minutes projected_minutes.csv \
  --team-games team_games.csv --opening-games 3 --output .
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

`NBA_SEASON` defaults to `2026-27`; `NBA_OPENING_GAMES` defaults to `3`.
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

The projection tests cover the three-game boundary, per-team counts versus player
appearances, unavailable post-window minutes, zeroes, missing statistical profiles,
rate blending, ambiguous names and rejection of preseason/stale-season data.
All 24 automated tests pass, including RotoWire fallback scaling, export parsing,
source precedence and explicit preservation of the remaining Hezonja gap. A clean Python 3.12 installation of requirements.txt
also passed Streamlit AppTest checks for full app startup, the four adjusted offensive
roles, workbook minutes, the 318-player positive-minutes view, the full 541-player
view, and team filtering. First-time graphics initialization required cache warmup.
Live NBA APIs and hosted deployment status were not tested.

## RotoWire fallback for missing historical profiles

The user requested RotoWire projections for players lacking a usable historical
statistical profile. `rotowire_fallback.py` normalizes an authorized CSV export,
matching NBA IDs or unambiguous player names and accepting explicit per-game,
season-total (requires GP), or per-36 units. Required core fields are Player,
MIN/MPG, PTS, REB, AST, STL, BLK, 3PM and TO. A RotoWire player ID is not an NBA ID.
Unknown players are reported; duplicates, missing core columns, invalid values and
zero rate denominators are rejected. Missing secondary stats are kept unavailable.

The app's RotoWire expander accepts a CSV for the current session; no account cookie,
password or uploaded subscriber file is stored in GitHub. An optional normalized
`data/rotowire_fallback.csv` can provide an explicitly supplied persistent baseline.
Only players without usable historical profiles use the RotoWire baseline. The
per-minute statistical rates still scale to the user's selected minutes. RotoWire
minutes never override workbook minutes in games 1–3 or season MPG from game 4.
From team game 5, current observations gradually replace the RotoWire prior using the same 200-minute
weight. All rows identify RotoWire as a forecast source, not actual prior-season stats.

An existing authorized RotoWire export was found in the user's Downloads directory:
`rotowire-nba-projections.csv`, modified 2026-10-16. It contains 457 projected player
rows with a two-line header and per-game fields. Normalized rates for **77 players
missing archived profiles** are stored in `data/rotowire_fallback.csv`; 30 of the 32
positive-minute players with missing baselines are covered. The original full
subscriber export and account credentials are not published. Source hash, date,
units and matching policy are recorded in `data/rotowire_source.json`. Name suffixes
can be omitted only when the remaining full name uniquely identifies one roster ID.

Pacôme Dadiet is absent from that RotoWire export but has actual 2025–26 regular-season
stats. A validated ESPN season summary (29 appearances) supplies his baseline through
`data/prior_stat_supplements.csv`, with separate source metadata. This does not invent
missing tracking features or change his approved archetypes.

With these sources, **317 of the 318 players with positive workbook minutes have
numerical core-stat previews** at zero current-season games. Mario Hezonja remains
unavailable: he lacks a 2025–26 NBA profile and is absent from the RotoWire export.
No older NBA season or unadjusted European profile has been silently substituted.
The live RotoWire full-season table stayed on “Loading Stats”, although the signed-in
Premium account could view individual forecast summaries. A complete usable updated
projection for Hezonja (including its minutes denominator) is still required.
