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

Selecting a player team defaults the opponent to its next verified regular-season fixture from the ESPN team schedule, retaining a current matchup until one hour after scheduled tip-off (the same window as Team Projections). Completed, postponed, canceled and wrong-season games are excluded. The opponent dropdown remains editable; changing players, minutes or sportsbook preserves a manual opponent choice, while choosing another team loads that team’s current/next opponent. If no fixture can be verified, the opponent starts blank for manual selection.

Automatic selections follow the schedule whenever the page reruns or you refresh live data; they are tracked separately from manual choices. **Use next opponent** (or clearing the dropdown) restores automatic selection. Only the schedule response is cached for five minutes, so an expired game cannot remain the automatic default just because it was cached. A recovered schedule fills an empty automatic selection. This does not add automatic page reloads or enable in-game sportsbook prices; retained matchups remain pregame model previews.

Sportsbook events for the default matchup must match its scheduled tip within 15 minutes. After tip, a later rematch between the same teams cannot supply replacement prices or team-total adjustments.

The **Player vs Opponent** tab selects a player and any other team, with an opponent-adjustment toggle and baseline/adjusted stat comparison. A 0–48 minute slider (half-minute steps) recalculates the selected player’s baseline and opponent-adjusted stat line from the underlying per-minute rates, including players whose automatic minutes are zero. Each player’s custom value persists within the browser session; **Use automatic minutes** restores the exact workbook/season-average value. Custom minutes affect only this player preview and its CSV, not the saved workbook or daily matchup rankings. The **Today’s Best Matchups** tab uses the day’s verified upcoming regular-season schedule (date selectable), ranks by any supported percentage, includes a negative-matchup toggle and minimum-sample filter, and displays a green/red heatmap, home/away, archetypes, sample counts, projected lines and CSV download. Empty slates stay empty; choose another date for previews. Rosters remain the user's workbook scenario, not confirmed lineups.

`matchup_projections.py` restores the legacy app's per-36 archetype-versus-opponent model. Only player-games with at least five minutes qualify. Each archetype's mean per-36 rate against an opponent is compared with its overall mean; deviations decay with half-lives of 15 samples for AST, 10 for OREB/DREB/STL/BLK and 999 for other stats, then shrink toward zero with three neutral samples. Offensive clusters drive PTS/AST/OREB/shooting/TOV; defensive clusters drive DREB/STL/BLK. Three-point makes use the attempts percentage to preserve individual conversion rates. REB = OREB + DREB and PRA = PTS + REB + AST. Sample counts are **player-games**, not unique team games. Missing/zero cluster baselines or missing matchup history remain blank, never an assumed neutral adjustment.

**Game-5 policy:** player stat-rate blending begins after the player's team completes four games; opponent tendencies use last season until the opponent completes four games, then use only current-season games. The minutes boundary remains game 4. This separates a player's rate sample from the opponent's defensive sample and lets teams cross the boundary independently. Current samples remain small, so the model retains neutral shrinkage and exposes counts. No pace, injury, starting-lineup or home-court adjustment is applied.

`data/game_logs_2026.csv.gz` contains 26,547 player appearances across all **1,230 2025–26 regular-season games**, retrieved 2026-10-19 from ESPN team schedules and summary box scores. Each of 30 teams has 82 games. The NBA Cup Championship is explicitly excluded even though ESPN tags it season type 2. Source details are in `data/game_logs_2026.csv.source.json`. Current approved archetypes/manual overrides label these historical appearances; archived labels cover players outside the current roster. Exact unambiguous canonical/workbook name matching links providers; unmatched players do not contribute cluster samples. Full-season averages derived from these box scores supersede partial archived stat baselines where matched; archived feature data remains unchanged for clustering. RotoWire still covers players without actual prior-season profiles.

Rebuild the historical snapshot with `python matchup_data.py --year 2026 --cache /path/to/local/boxscore-cache`. The cache holds raw source JSON outside the repository. Current-season schedules/box scores refresh every five minutes once a team reaches the matchup boundary; completed box scores are cached locally to avoid downloading them again. Every box score must match the requested season, type and completed status. Partial feed failures produce a visible warning.

### Sportsbook comparison and juice-adjusted totals

Player vs Opponent also includes **PF (personal fouls committed)** with a baseline,
matchup rating and projection. Foul rates use the same projected minutes (including
the manual slider) and game-5 season blend as the other stats. The defensive
archetype's per-36 foul tendency against the opponent supplies the matchup effect,
with a 10-sample half-life and the existing three-sample neutral shrinkage.
Fewer projected fouls produce a better matchup rating. Turning matchup adjustments
off shows the minutes-scaled baseline. PF does not change PRA or receive team-market
calibration, and is not added to Top Sportsbook Edges or its stat filters.

The 2025–26 box-score archive now includes PF for all 26,547 player appearances,
backfilled from the same cached ESPN source responses without changing any other
column. Current summaries read ESPN's `avgFouls`; current box scores read `PF`.
Optional RotoWire `PF`/`FOULS`/`PERSONAL FOULS` columns are supported. The stored
RotoWire fallback does not contain PF, so players relying solely on it have blank
foul projections until a valid foul rate is supplied or current-season rates become
eligible. PF is optional and cannot block other projections; when only one eligible
prior/current foul rate exists, that rate is used. Missing matchup foul samples also
stay blank. No sportsbook foul market is connected, so PF book/edge cells stay blank.

The player projection table now includes **Book line**, **Book implied total**, and **Edge %** and **Play** for PTS/REB/AST/3PM/STL/BLK/TOV/PRA. Select FanDuel (default), DraftKings, or BetMGM. Percentage edge equals the absolute value of `(Projected / Book implied total - 1) × 100`, rounded to two decimals: both Over and Under edges display as positive. A gap strictly greater than 5% receives an Over label and green shading when the projection is higher, or an Under label and red shading when lower. Exactly 5%, smaller edges, zero/missing implied totals and missing projections remain unmarked. Color intensity increases continuously from pale at 5% to saturated at 25% (clipped thereafter). Only the Edge % and Play cells are highlighted; projected totals are plain. The 5% rule is an extra projection buffer on top of the already no-vig model estimate, not a calculated wagering ROI. This is a numerical projection comparison, not an EV calculation or bet recommendation. Both projection and top-edge comparison downloads retain **Signed edge %** alongside the positive **Edge %** for direction, including rows below the play threshold.

The existing The Odds API access is reused. Prefer `ODDS_API_KEY` in environment variables or Streamlit secrets; the existing legacy feed configuration remains compatible. Credentials are not displayed or included in request-error messages. Upcoming NBA events are cached for five minutes; only the next future event matching the selected team/opponent is queried for the chosen bookmaker's eight standard prop markets. A real scheduled opponent is required; arbitrary matchup previews may have no book lines. Per-event props are also cached for five minutes, limiting requests to the selected event/book, rather than loading every NBA game on every rerun. “Refresh live data” clears both caches.

Only exact normalized player matches and paired over/under prices on the **same book, game and posted line** are used. Different alternate lines, ambiguous/duplicated prices and unmatched player names are not combined. Timestamped quotes more than 15 minutes old (or without usable timestamps) cannot drive a colored comparison. Missing categories show blank comparisons rather than estimated or copied betting lines. The detail expander shows posted line, over/under odds, fair over probability, update time and availability status, with a CSV download.

American prices give raw implied probabilities: negative `a` gives `abs(a)/(abs(a)+100)`; positive `a` gives `100/(a+100)`. Remove margin proportionally: `p_over_fair = p_over_raw / (p_over_raw + p_under_raw)`. Solve for a **Poisson-model mean** such that `P(X > floor(line)) = p_over_fair`. Integer lines instead use `P(X > line | X != line)` so pushes are handled correctly. This model-implied mean is not a sportsbook-published total; a single line and two prices do not uniquely determine the true statistical distribution. Poisson's variance assumption is a simplifying approximation, particularly for points and combined stats.

Illustrative example, not a current Brunson quote: **over 2.5 threes -150 / under +120 → fair over probability 56.90% → implied mean approximately 2.965**, above 2.5. The under price is required to remove margin; -150 alone cannot determine a no-vig mean. Manual minutes and the opponent-adjustment toggle change the app projection, while the sportsbook implied total stays tied to its actual event and prices. Existing archetypes, override precedence, game-3 minutes and game-5 stat rules are unchanged.

Feed documentation: [The Odds API v4 event odds](https://the-odds-api.com/liveapi/guides/v4/), [supported NBA prop markets](https://the-odds-api.com/sports-odds-data/betting-markets.html#nba-ncaab-wnba-player-props-api).

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
  10 projected minutes**, **259 had usable feature profiles for named-role assignment** and **26 lack a saved profile and now have approved manual assignments**.
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
  hashes, workbook provenance, default assumptions and named-reference model policy.

`SEASON_ID=22026` is the assignment season. `SourceSeasonID=22025` identifies observed
feature provenance. `FeatureDataStatus=preseason_historical_prior` marks reused
profiles. Missing profiles have no source season. `AssignmentSource` distinguishes
`historical_named_role`, `named_reference_centroid`, `analyst_review`,
`manual_override`, `outside_minutes_cutoff` and `missing_historical_profile`. Eligibility and profile availability are separate flags.
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

The former KMeans/Hungarian alignment has been removed from the validated season runner. It forced dissimilar newly fitted groups onto historical IDs and allowed an old one-player outlier category to absorb 44 players. The original defensive name dictionary also omitted four historical IDs. Numeric cluster IDs were therefore being mistaken for stable basketball roles.

`archetype_model.py` now uses the original 34 offensive and 13 defensive features with **fixed, named reference roles**. In preseason prior mode, the same archived player features keep their named historical/reviewed role; changing roster composition or the projected-minutes cutoff does not refit or permute roles. For genuinely new/current feature profiles, standardized distances to fixed named reference centroids provide a role estimate. Centroids use observed profiles with at least 10 MPG and five archived appearances; invalid/unnamed labels cannot be anchors. Automatic guard-sized profiles cannot be assigned center-only categories; explicit human rules can authorize exceptions. No one-to-one matching, requirement to populate every class, or arbitrary raw cluster number is used. These are role estimates, not a guarantee of perfect basketball classification.

Precedence (highest first): **existing user manual override → analyst role review → named historical role (preseason) or fixed named centroid (new features)**. `archetype_overrides.json` is unchanged by the repair. `Uncategorized` remains only a legacy dictionary entry and is ineligible for automatic assignment. Undefined defensive IDs 2/9/10/13 cannot be published. Validation blocks exports and live loading containing unnamed assigned roles. Genuinely missing profiles and players outside the existing >10-minute model cutoff may remain unassigned when neither a named historical role nor an explicit human review is available; they are not silently given invented profiles or catch-all roles.

Observed counts remain divided by observed `MIN` for per-36 features. Projected minutes do not replace that denominator or create fake observations. Full-season statistical-rate projections, RotoWire fallbacks and the game-3/game-5 timing rules are unchanged by this role repair.

### Complete role audit (2026-10-19)

`data/audit/affected_players_before.csv` freezes the original **100 distinct players**: 44 with offensive `Uncategorized`, 62 with unnamed defense, and six in both lists. `archetype_reviews.csv` records an offensive and defensive decision, role rationale, source and confidence for every one of the 100. All have named roles in the rebuilt roster. The audit includes scoring/creation roles, off-ball shooters/cutters, guard versus wing defense, interior defenders and existing manually assigned exceptions; this is a player-by-player semantic review, not a cosmetic renaming of whole mixed clusters.

The same defective fit had changed already-named assignments for 90 other players. These were also checked in `archetype_reference_reviews.csv` while rebuilding the semantic anchors. Examples include restoring a perimeter role for Derrick White, an interior finishing role for Aaron Gordon and rim protection for Victor Wembanyama. All preexisting user overrides remain authoritative. These additional judgments are separately recorded from the requested 100-player audit.

Sources are the immutable archived 2025–26 tracking/shot-profile snapshots and player-role interpretation within the existing archetype vocabulary. These snapshots are partial-season evidence, not newly observed 2026–27 games. Developing or small-sample roles are flagged provisional. The legacy `Spot-Up Role Player` bucket is used for low-usage complementary roles, including cutters; no new archetype vocabulary was invented. All per-player reasons and archived GP/MPG/per-36 evidence are downloadable in `data/audit/reviewed_100_players.csv` and the live **Archetypes → Review the 100 corrected players** expander. `data/audit/additional_90_players.csv` records the additional semantic review. `data/audit/all_assignment_changes.csv` lists actual before/after role changes across the roster, and `review_manifest.json` records source hashes and protected-rule counts.

Reviewed roles also relabel historical opponent samples. Historical players outside the current roster and profiled players below the minutes cutoff are mapped to named reference roles for matchup sample construction; old unnamed IDs cannot survive as hidden statistical groups. Current roster assignments win, and all user rules are reapplied at the live-app boundary. Cached matchup effects include the review/rules/model hash so role changes invalidate stale results.

### Additional 33 rotation players (2026-10-19)

The user's 33-player list was missing one or both roles because all were at or below the old strict >10-minute training cutoff. `archetype_rotation_reviews.csv` now records per-player offensive/defensive assignments and reasons. Kyle Anderson's existing Non-POA Wing defensive override remains authoritative. All **318 players with positive workbook minutes** now have both roles, including all 33 listed players; the other players' roles and every workbook minute are unchanged by this addition.

Training eligibility remains separate from role availability: the >10-minute cutoff controls model inputs, but positive-minute players with real archived profiles can retain named reference roles below it. Reviewed assignments cover missing-profile cases without inventing tracking data or statistical observations. The six listed players without archived feature profiles retain that missing-profile status; their role labels are provisional judgments, not observed 2026–27 statistics. Sources include archived 2025–26 profiles plus [Dailyn Swain](https://www.nba.com/draft/2026/prospects/dailyn-swain), [Meleek Thomas](https://www.nba.com/draft/2026/prospects/meleek-thomas), [Allen Graves](https://www.nba.com/draft/2026/prospects/allen-graves), [Kobe Sanders](https://www.nba.com/draft/2025/prospects/kobe-sanders) and [Yanic Konan Niederhauser](https://www.nba.com/draft/2025/prospects/yanic-konan-niederhauser) official prospect profiles. Role interpretations are recorded in the CSV rather than treated as official NBA archetype labels.

`data/audit/low_minute_33_players.csv` and the **Archetypes → Review the 33 additional rotation players** expander provide the complete assignments and download. Run `python audit_archetypes.py` after rebuilding to regenerate this file from the review and roster IDs. User manual overrides remain above all analyst reviews. Existing projection rates, custom minutes, sportsbook implied totals and edge filters are unchanged.

## Run or refresh

```sh
python -m pip install -r requirements.txt
python import_minutes.py /path/to/Custom_Team_Minutes_2026-10-18.xlsx --output data/preseason
python preseason_bootstrap.py --minutes data/preseason/projected_minutes.csv --output . --opening-games 3
python audit_archetypes.py
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
numerical core-stat previews** at zero current-season games. Mario Hezonja lacks
a 2025–26 NBA profile and is absent from the RotoWire export. The October 24
archetype-average fallback below now supplies an explicitly provisional estimate.
No older NBA season or unadjusted European profile has been silently substituted.
The live RotoWire full-season table stayed on “Loading Stats”, although the signed-in
Premium account could view individual forecast summaries. A complete usable updated
projection for Hezonja (including its minutes denominator) is still required.


### User corrections and read-only fit audit (2026-10-19)

Ausar Thompson and Jrue Holiday are now **Secondary Playmakers** on offense,
recorded as user manual overrides. Their defensive roles are unchanged. All 120
previous manual rules retain their exact values; there are now 30 offensive and
92 defensive rules. Only those two players' published role assignments changed.

`python audit_role_fit.py` screens all 318 players with positive workbook minutes
on both sides (636 checks). It does not change any assignment, review or override.
The screen uses the existing 34 offensive/13 defensive standardized archived
2025–26 tracking features and fixed named centroids, respecting the automatic
height guardrail. Flags identify top-two distance margins below 10%, an assigned
role at least 35% farther than the nearest centroid, nearest distance above the
reference 95th percentile, fewer than 10 archived games/MPG, or absent features.
These are review thresholds, **not calibrated confidence probabilities or proof
that a label is wrong**. Partial-season tracking data can miss changed roles;
steals, blocks, size and centroid proximity do not establish defensive assignments.

`data/audit/role_fit/` contains every check, every flagged side and source hashes.
The broad screen flags 209 players (274 sides), including 83 protected manual
sides. A flag on a manual rule is informational; the rule remains authoritative.
`data/audit/role_fit_proposals.csv` records analyst review of the 28 strongest
nonmanual centroid conflicts plus three role/position concerns. Six changes are
proposed for user approval; the other 25 reviewed sides should retain their
current labels. Other weak, missing-data or borderline flags retain current roles
pending stronger evidence. **No proposal in this file is loaded by the app or
clustering pipeline.** Neither the audit nor its proposals modify live labels.
The proposed changes may only be incorporated after explicit user approval.


### Accepted audit decisions (2026-10-19)

The user approved eight role changes: Jock Landale → Stretch Big (offense),
Caleb Martin → Spot-Up Role Player (offense), Noah Clowney → Catch & Shoot Wing
(offense), Daniel Gafford and Walker Kessler → Rim Protector (defense),
Ryan Rollins → POA Defender (defense), Tristan da Silva → Long Wing/Secondary POA
(defense), and Collin Gillespie → Pesky Guard (defense). Nae’Qwan Tomlin retains
Athletic Roll Man, now explicitly protected as a manual offensive override.
All 122 previously stored rules are preserved; totals are now 34 offensive and
97 defensive manual rules. No other assignments or source minutes changed.
`data/audit/user_accepted_role_changes.csv` records the eight actual changes.
The proposal ledger retains the original assignments/rationales for history and
marks these decisions applied; approved user decisions supersede earlier analyst
recommendations. The audit validates approved roles against manual rules and
never reapplies a stale candidate. The refreshed screen has 204 flagged players /
268 sides, including 86 protected manual sides, with zero pending role changes.
The earlier 209-player counts above describe the initial pre-approval screen.


### Top 20 sportsbook edges per game

The **💰 Top Sportsbook Edges** tab provides one-click game buttons plus date,
sportsbook (FanDuel/DraftKings/BetMGM), and Over/Under filters. It defaults to the
next date offered by the sportsbook feed and shows both teams' top 20 distinct
player/stat comparisons, ranked by the magnitude of the percent gap between the
opponent-adjusted projection and the no-vig, Poisson-implied book total. This is
not a probability edge or expected return. Both sides' American prices and the
posted line remain visible. Multiple props for one player can be correlated.

**Filter by player** accepts a full or partial name (case and accents are ignored).
Press Enter to apply it; clear it to restore all players in the selected game.
Search applies before the top-20 limit, so it can find players outside the
unfiltered top 20. Stat, direction and 5% filters still apply, and the CSV matches
the filtered results. Team adjustments always use the full roster before search;
filtering does not change anyone's projections.

Edge % is positive for both Overs and Unders; the Play label gives the direction.
The >5% play filter is on by default; exactly 5% does not qualify. Disable it to
view the strongest remaining comparisons without marking them as plays. Existing
continuous edge colors apply. If fewer than 20 props qualify, only real eligible
rows appear. CSV downloads include quote timestamps and matchup sample coverage.
Game buttons include future sportsbook events that match the selected date's
ESPN regular-season schedule on home/away teams and tip time (within 15
minutes). The tolerance handles the difference between an advertised start and
the book's updated expected tip, such as PHI–NYK's 5:00 PM versus 5:09 PM MDT on
October 20. Team Projections uses the same tolerance when attaching market lines.
Only ESPN games still marked Scheduled qualify; in-progress, final, postponed,
canceled, suspended or delayed events and past sportsbook tips are excluded.
There is no fallback to a different game, book, one-sided line, stale quote, or
invented projection. Both paired prices must be current within 15 minutes.

Rankings use automatic rounded minutes (workbook for the first three team games,
season MPG thereafter), the existing game-five stat/matchup policy, and approved
manual archetype overrides. The individual player's minutes slider still affects
that preview only. Zero/unavailable minutes, missing statistical baselines or
incomplete matchup history exclude a player from this table; the coverage
expander explains the underlying inputs. Props are loaded only for the selected
edge game, cached for five minutes, and refreshed by the existing refresh button.


### Stable browsing and explicit refresh

The dashboard no longer runs a timed five-minute fragment rerender, which rebuilt
all tabs and tables while a user was reading or scrolling. The fragment now reruns
on user controls only. **Refresh live data** clears the five-minute caches and
reloads records, schedules, matchup history and sportsbook quotes. Changing game,
book or filters still updates that selection; cached data expires after five
minutes when a subsequent interaction requests it. The visible checked-at time
identifies the snapshot. It stays fixed during passive browsing, and quote-age
validation still excludes prices older than 15 minutes at comparison load time.


The Top Sportsbook Edges page includes persistent checkboxes for PTS, REB, AST,
3PM, STL, BLK, TOV and PRA (all enabled initially). Stat selection applies before
ranking and the 20-row limit, so the table and CSV contain the strongest edges
within the checked categories. Unchecking every category shows a selection prompt.


### Over-price labels and matchup ratings

Player vs Opponent shows **Book line (Over)** as the original posted line plus
its signed American Over price, e.g. `1.5 -150` or `2.5 +120`. The top-edges table
uses the same label. Numeric line and both prices remain available in exports;
no-vig implied totals and percentage edges use the original numeric inputs.

Between Baseline and Projected, **Matchup rating** measures opponent-adjusted
production relative to that same minutes-adjusted baseline: 1 ⬇️⬇️ for ≤−10%,
2 ⬇️ for (−10%,−3%), 3 ↔️ for [−3%,+3%], 4 ⬆️ for (+3%,+10%), and 5 ⬆️⬆️
for ≥+10%. Turnovers invert the sign so fewer turnovers rate higher. Rebounds
and PRA use their combined baseline/projected totals. These explicit display
thresholds are an interpretive scale, not calibrated outcome probabilities or
sportsbook edges. Ratings remain blank when matchup adjustments are disabled,
baseline is zero/nonfinite or projected data is unavailable.


Top Sportsbook Edges excludes turnovers (TOV) from both its category checkboxes
and rankings. Select all selects the seven remaining categories. Player vs
Opponent retains its existing turnover projection.


### Full market team-total matching: points only

Player vs Opponent and Top Sportsbook Edges apply the selected book's posted
spread and game total to **PTS only**, after opponent adjustments. Home implied
points = (total − home spread)/2; away = (total + home spread)/2. Each player's
PTS is multiplied by market team points / the full positive-minute team's model
PTS. There is no partial blend. No intermediate rounding is used, so each team
matches the market total before display rounding. Original point shares remain
intact. This points step does not scale other stats. The team assist adjustment
below then updates AST, and the rebound step updates REB. Final PRA =
market-adjusted PTS + updated REB + updated AST. Points, assists, threes, rebounds
and PRA edges/ranks use their final projections.
Matchup ratings retain the original opponent-only effects before team adjustments.

Fresh same-event, same-book paired spreads/totals are required; posted game
lines are used directly, not a juice-derived game mean. Quotes older than 15
minutes, mismatched/ambiguous lines and started games cannot calibrate points.
Missing markets or incomplete team point/minutes coverage leave that team's
original estimate intact and show an explanatory caption. The other team's valid
calibration can still apply. Game markets share the five-minute cache and explicit
refresh policy. The player slider updates that player's contribution in the full
team denominator before scaling; other session slider values do not affect this
preview or the automatic-minute edge rankings. Disabling opponent adjustments
uses the unadjusted team baseline as the denominator; market matching still applies.

### Team assists with normalized archetype shares

Both Player vs Opponent and Top Sportsbook Edges use the same assist calculation
after points calibration. Historical rates use all player appearances across the
validated 1,230-game **2025–26 regular season**, with 82 games per team, from
`data/game_logs_2026.csv.gz`. Team totals include appearances shorter than five
minutes and players without archetype assignments. Rates are ratios of season
totals, not averages of individual percentages. Historical team/opponent records
are used, independently of current workbook team membership and manual overrides.

* Expected assisted-basket rate = 50% × team AST/FGM + 50% × opponent AST
  allowed/FGM allowed.
* Expected team FGM = selected sportsbook's implied team points / [50% × team
  PTS/FGM + 50% × opponent PTS allowed/FGM allowed]. The total-points-to-FGM ratio
  includes free throws and the mix of two- and three-point baskets. This is an
  internal estimate shared by the assist and three-point team targets.
* Team AST target = expected assisted-basket rate × expected team FGM.
* Player AST = team AST target × existing opponent-adjusted player AST / sum of
  existing opponent-adjusted AST for the full positive-minute roster.

Thus player assists sum exactly to the team target before display rounding.
Shared defensive multipliers cancel in the normalized shares; relative archetype
differences remain. Equivalently, each archetype modifier is divided by the team's
baseline-assist-weighted average modifier. No second archetype multiplier is
applied after allocation. Custom minutes update the selected player's production
and share in the full roster; the edges page retains automatic minutes. With the
opponent-adjustment toggle off, the allocation uses unadjusted player shares.

The 50/50 blends and last season's team styles are explicit modeling assumptions,
not sportsbook-published assist totals or optimized weights. Team-target rates
remain on 2025–26 until that policy is revised; the existing game-5 player-rate
and opponent-archetype policies still govern the allocation weights. Roster
turnover can make last year's team style less representative.

Fresh verified game lines are required, using the existing book/event/freshness
checks. Missing rates, invalid minutes, or incomplete positive-minute AST/PTS/REB
coverage retain that team's existing assist estimate, with an explanatory caption;
another team's valid adjustment can still apply. Zero-minute players receive no
allocation. The assist step leaves shooting and other stats unchanged; threes are
calibrated separately below. PRA always combines final PTS + REB + AST. Projection and edge CSVs retain the original
AST, team target, allocation factor, expected FGM and player share for auditing.

### Team threes with normalized archetype effects

Player vs Opponent and Top Sportsbook Edges apply the reviewed team-first model
after points and assists. All 2025–26 regular-season appearances contribute to
historical ratios; the same paired-game and 82-games-per-team checks apply.

* Expected team FGM uses the same market team points / blended PTS-to-FGM estimate
  as assists.
* Team 3PM target = expected team FGM × [50% × historical team 3PM/FGM + 50% ×
  historical opponent 3PM allowed/FGM allowed]. These ratios describe the share
  of **made baskets** from three, not three-point shooting percentage.
* Player 3PM = team 3PM target × current minute/archetype-adjusted player 3PM /
  the full positive-minute roster's sum of those projections.

This is equivalent to allocating the team target using pre-matchup shooting
shares, then multiplying each player's archetype effect by the inverse of the
team's baseline-three-weighted average archetype modifier. Shared defensive
effects cancel, relative player effects remain, and final player threes sum to
the target before rounding. The 50/50 blend and prior-season team style remain
explicit modeling assumptions. Team rates stay on 2025–26; existing player and
archetype rate updates still change the allocation weights.

Supporting FG3A scales by the same factor as FG3M to preserve player conversion
rates. This changes the shooting mix within the existing points estimate:
**PTS, AST, REB and PRA are not increased or decreased again for the threes
change**. Matchup ratings retain pre-calibration effects, while 3PM sportsbook
edges/ranks use calibrated makes. CSVs preserve original makes/attempts, the team
target and factor, historical rate and player share.

Fresh matching game lines and complete positive-minute FG3M/FG3A/PTS coverage are
required. Invalid or missing minutes, impossible shooting rates, or a result where
three-point scoring would exceed any player's total points preserve that team's
existing threes with a visible explanation. Another team's valid calculation may
still apply. Zero-minute players receive no allocation; all manual overrides are
preserved.

**Minutes changes:** no player share or normalization factor is cached. When
updated team minutes are imported, published to the app's roster snapshot, and
the app reloads, both pages recalculate their player projections and shares. The
team target remains tied to market points and historical matchup rates. A local
spreadsheet edit alone does not synchronize with Streamlit. The individual slider
recalculates that player's preview against the rest of the team's automatic
minutes; it does not overwrite the saved team allocation used by the edges page.

### Team Projections tab

Select a date, scheduled game and sportsbook to compare both teams' final PTS,
REB, AST, 3PM, STL, BLK, TOV and PRA, alongside total projected minutes. A second
table traces PTS/AST/3PM/REB from baseline to archetype-adjusted player sums, then the
team target and final total. The adjustment percentage is final divided by the
archetype-adjusted sum minus one. Targets are displayed once per team, never
summed across player rows.

This tab uses the same full-roster projections, saved automatic minutes and
points → assists → threes → rebounds calculations as Top Sportsbook Edges. It introduces
no new projection model and does not use individual player-slider overrides.
Player contributions and target-method explanations are expandable; team totals
and the adjustment breakdown can be downloaded as CSVs. Totals use unrounded
player values and may differ slightly from adding rounded display values.

Games come from the ESPN regular-season schedule, even if no sportsbook
event is available. Team Projections retains games, including in-progress games,
until one hour after the ESPN scheduled tip (exclusive cutoff); canceled,
postponed, delayed, suspended or final games remain excluded. The list updates
on user interaction/refresh without adding an automatic rerun. Top Sportsbook
Edges retains its existing pregame-only eligibility.

When a pregame market-adjusted team projection is loaded, this session retains
the baseline, matchup and final player tables separately for that game and book.
After tip, if pregame lines are unavailable, Team Projections shows that saved
projection with its timestamp rather than using in-game odds. Snapshots expire
at the same one-hour cutoff and do not persist across sessions; without one,
the existing model-only fallback is clearly labeled.

Team calibration requires the same verified event, selected
book and fresh spread/total as the other pages. If unavailable, existing model
estimates remain visible with blank targets and an explanation. Missing minutes
blank the whole team's totals; a missing positive-minute player's stat blanks
that stat's total rather than displaying a misleading partial sum. Zero-minute
players are excluded. The existing explicit Refresh live data button updates
sources; no timed refresh or scroll-triggered rerun is added.

### Team rebounds with normalized offensive/defensive shares

The approved BOS–DET / PHI–NYK preview model now runs in Player vs Opponent,
Top Sportsbook Edges and Team Projections. Team Projections includes REB in its
adjustment breakdown, OREB/DREB in player contributions, and a separate expandable
table of rebound targets and opportunity pools. The historical input is the same
full 2025–26 regular-season player box-score archive used above; all appearances
count, with paired opponents and 82 games per team required.

For each shooting team, independently of the current roster:

1. Expected FGM = market team points / mean(team PTS/FGM, opponent allowed
   PTS/FGM), matching the existing assist/three-point calculation.
2. Expected FG misses = expected FGM × mean(team (FGA−FGM)/FGM, opponent
   allowed equivalent). Expected FT misses use (FTA−FTM)/FGM in the same way.
3. Credited player rebound pool = expected total misses × mean(team credited
   fraction, opponent allowed credited fraction). The historical fraction is
   (shooting team's player OREB + opposing player DREB) / (FG misses + FT misses).
4. Offensive capture share = mean(team OREB / its credited pool, opponent's
   OREB allowed / its allowed credited pool). The shooting team's OREB target
   is pool × share; the defending team's DREB target is pool × (1−share).
   These complementary targets conserve each pool exactly.
5. Allocate each team's OREB and DREB **separately**, using target × player's
   existing minute/archetype-adjusted component / team sum of that component.
   This equals applying individual archetype modifiers divided by the team's
   baseline-component-weighted modifier. Common effects cancel; relative
   offensive and defensive archetype differences remain.

Final REB = adjusted OREB + adjusted DREB; final PRA = existing calibrated PTS +
adjusted REB + existing calibrated AST. Points, assists, shooting, steals,
blocks, turnovers, minutes and manual archetypes are unchanged by this step.
Ratings keep the original opponent-only REB/PRA effects; sportsbook projections,
edges and ranks use the updated totals. Downloads retain original components,
targets, factors, player shares and the opportunity inputs.

The 50/50 blends and prior-season team styles are explicit assumptions, not
backtested weights. The credited fraction is a box-score approximation for
non-reboundable free throws and dead-ball/team rebounds, not tracking or
play-by-play rebound chances. There are no sportsbook rebound-total inputs.
Expected misses are internal target inputs, not changes to the player shooting
projections. Historical team rates stay on 2025–26; the existing game-5 policies
still update individual allocation weights.

Both fresh, verified market team point totals are required, including in a
single-team player preview, because defensive rebounds come from the opponent's
misses. Missing rates, invalid minutes or incomplete positive-minute
OREB/DREB/REB/PTS/AST coverage retain that team's existing rebounds and PRA
atomically with an explanation. Another team's valid allocation can still run.
Zero-minute players receive no allocation. Minutes changes recalculate the
normalized shares without changing a target for fixed game lines. The existing
explicit refresh policy and workbook import/publish requirements still apply.


### Game-specific approved minutes (October 20, 2026)

Use **Import Approved Minutes** near the top of the live dashboard. Upload the
JSON produced by the NBA model’s **Download Approved Minutes** button after
explicitly approving both teams in every game on that day. The validated file is
saved in this browser's local storage and restored after refresh, reopening the
app, or an app reboot. Another browser/profile needs its own import; already-open
tabs pick up the saved file on their next full refresh. The file is never committed,
publicly shared, or written over the baseline workbook. No new credentials or
shared data store are required. A visible banner names the active date and source;
the team and sportsbook pages also show each game's minute sources and totals.

The importer checks the bundle checksum, season/date, pairings, two approvals
per game, unique player IDs, 240 minutes and five starters per team, and roster
coverage for all players receiving minutes. Extra zero-minute NBA roster rows
are allowed; prop-only players on an approved team receive zero. Unknown active
players block the import with a specific name/ID. A checksum detects accidental
edits; it is not a cryptographic identity/signature or a live approval check.
After import, active players missing stat baselines or archetypes are listed
under **Players needing projection data**. Unclassified players prefer available
RotoWire rates scaled to approved minutes; if no RotoWire row exists, a valid
historical baseline is retained. An unassigned archetype side retains baseline
production without an opponent multiplier. Known archetype sides still receive
their usual effects. These contributions remain in team calibration and normalized
shares; no player is silently dropped. Missing *statistical* baselines without
a supported, flagged archetype estimate still block complete team totals. Unclassified players have no matchup rating and do not enter
the top-edge rankings. Importing minutes never creates or overwrites archetypes.

The October 20 fallback update adds seven rows from the already authorized
October 16 RotoWire export. Source hashes and covered player IDs are recorded in
`data/rotowire_source.json`. A zero-minute or missing RotoWire projection cannot
supply per-minute rates; it is never treated as a prediction of zero production
at positive approved minutes. Private user-amended rotations retain the source
bundle and explicit adjustment provenance and remain outside the repository.

Approved minutes apply only to the exact Eastern-calendar-date matchup and
supersede workbook/season-average minutes, preserving exact decimals. Stat rates,
injury policy, archetypes and market formulas are unchanged. All four projection
views use the imported allocation; manual player-preview sliders still work. A
new import clears previous slider edits and cached team snapshots. Removing an
uploaded file or clicking **Clear saved minutes** clears the browser copy and
restores the automatic baseline. Restored data goes through the same complete
validation as uploads. Projections wait for restoration before rendering; invalid
saved/replacement data pauses projections rather than silently ranking props with
old minutes. If browser storage is blocked, a visible warning explains that an
upload will only last for that session. A file cannot detect subsequent changes
in the NBA model: download and import a fresh file after reapproving.

Validation: `../app-check-env/bin/python -m unittest discover -s tests -q`.
Browser-storage behavior: `node --test tests/test_approved_minutes_storage.mjs`.


## Temporary archetype-average statistical profiles (October 24, 2026)

When a player has no usable historical or RotoWire profile and current-season
stats are not yet eligible, `archetype_fallback.py` supplies an explicitly labelled
estimate from players with the same assigned archetypes. Offensive roles supply
PTS, AST, OREB, shooting and TOV; defensive roles supply DREB, STL, BLK and PF.
Each donor contributes equally to the mean of historical per-minute rates. A side
requires at least three distinct historical players with a complete, finite,
nonnegative stat vector. Forecasts and synthetic profiles are excluded as donors.
Both sides must qualify; no roles or observations are invented. REB = OREB + DREB
and PRA = PTS + REB + AST. The approved minutes stay unchanged.

Priors are built once from the league roster and reused in all game and individual
previews, so filtering the displayed team or player cannot change the estimate.
Individual historical/RotoWire profiles take precedence. Once current-season
stats become eligible under the existing team-game-five rule, a player without
an individual prior uses those observations directly; the synthetic estimate is
never blended into or saved as historical evidence. Other players' profiles and
the existing team adjustment formulas are unchanged.

The player row and detailed preview identify the temporary baseline and donor
counts. Team totals and their CSV carry `Stat coverage` and `Estimated players`;
a visible warning names every active estimated player and their combined minutes.
Approved-import coverage also flags the missing individual profile. Remaining
unresolved gaps name the affected players and fields and still leave totals blank.
Zero-minute players do not make a team provisional. Estimated players remain
excluded from top sportsbook-edge rankings until individual stats are available.

Cleveland's Mario Hezonja lacks an individual statistical profile, which previously
left all eight team totals blank when he received minutes. His assigned Catch &
Shoot Wing and Non-POA Wing roles now provide a temporary estimate; other players
retain their original individual baseline profiles. Regression coverage uses a
synthetic 240-minute rotation with Hezonja at 20 minutes against Brooklyn.
The existing points, assist, three-point and rebound team adjustments can then
calibrate the complete rotation. No saved minutes, archetypes or source stats are
changed. This release publishes the previously local fallback to the hosted app.

## Responsive projection tabs (October 26, 2026)

Each tab now runs in its own Streamlit fragment. Changing a player, opponent,
sportsbook or minutes slider recalculates only Player vs Opponent. It reuses the
loaded league records, statistical baselines and matchup history rather than
starting another league-wide download when the five-minute data cache expires.
Filters in the other tabs likewise rerun only their own view. The tab layout,
projection formulas, manual archetypes and approved-minutes precedence are unchanged.

**Refresh live data** still clears the live feed caches and rebuilds all views.
Importing or clearing minutes, or changing a RotoWire upload, rebuilds the shared
snapshot and game projections. Selected-game schedule and sportsbook calls retain
their five-minute cache and normal event/quote-age checks; a new game or expired
feed cache can still require a short network wait. Initial loading and an explicit
refresh still load the full dashboard.

`tests/test_dashboard_fragments.py` exercises actual fragment-scoped reruns with
external feeds stubbed: player changes do not refetch league data or execute hidden
tabs, manual minutes still recalculate and reset, Refresh rebuilds all tabs, and a
replacement approved import changes the game projections instead of reusing old
minutes. The test runner supplies fragment IDs because Streamlit 1.53 AppTest
normally sends full-script reruns.

## Schedule recovery (October 27, 2026)

Daily ESPN scoreboard requests retry once after a network or invalid-response
failure. Responses must contain a valid event list, the requested season and
Eastern schedule date, and complete regular-season matchups. Late games after
midnight UTC retain their correct US schedule date. Verified empty slates remain
empty; failed requests no longer display the misleading "No upcoming games"
message.

Today's Best Matchups, Team Projections and Top Sportsbook Edges each provide a
**Retry schedule** button on failure and **Reload schedule** after success. These
clear only the selected date's schedule cache and rerun the current tab, preserving
the loaded player data and approved minutes. The existing five-minute schedule
cache, regular-season filtering and game-retention rules still apply.

`tests/test_schedule_feed.py` checks bounded retries, malformed and mismatched
responses, late games, and legitimate empty/completed slates. The dashboard
fragment test verifies recovery without a league-wide reload or a false empty-slate
message.
