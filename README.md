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

Selecting a player team defaults the opponent to its next verified future regular-season fixture from the ESPN team schedule. Completed, postponed, canceled and wrong-season games are excluded. The opponent dropdown remains editable; changing players, minutes or sportsbook preserves a manual opponent choice, while choosing another team loads that team’s next opponent. If no fixture can be verified, the opponent starts blank for manual selection.

The **Player vs Opponent** tab selects a player and any other team, with an opponent-adjustment toggle and baseline/adjusted stat comparison. A 0–48 minute slider (half-minute steps) recalculates the selected player’s baseline and opponent-adjusted stat line from the underlying per-minute rates, including players whose automatic minutes are zero. Each player’s custom value persists within the browser session; **Use automatic minutes** restores the exact workbook/season-average value. Custom minutes affect only this player preview and its CSV, not the saved workbook or daily matchup rankings. The **Today’s Best Matchups** tab uses the day’s verified upcoming regular-season schedule (date selectable), ranks by any supported percentage, includes a negative-matchup toggle and minimum-sample filter, and displays a green/red heatmap, home/away, archetypes, sample counts, projected lines and CSV download. Empty slates stay empty; choose another date for previews. Rosters remain the user's workbook scenario, not confirmed lineups.

`matchup_projections.py` restores the legacy app's per-36 archetype-versus-opponent model. Only player-games with at least five minutes qualify. Each archetype's mean per-36 rate against an opponent is compared with its overall mean; deviations decay with half-lives of 15 samples for AST, 10 for OREB/DREB/STL/BLK and 999 for other stats, then shrink toward zero with three neutral samples. Offensive clusters drive PTS/AST/OREB/shooting/TOV; defensive clusters drive DREB/STL/BLK. Three-point makes use the attempts percentage to preserve individual conversion rates. REB = OREB + DREB and PRA = PTS + REB + AST. Sample counts are **player-games**, not unique team games. Missing/zero cluster baselines or missing matchup history remain blank, never an assumed neutral adjustment.

**Game-5 policy:** player stat-rate blending begins after the player's team completes four games; opponent tendencies use last season until the opponent completes four games, then use only current-season games. The minutes boundary remains game 4. This separates a player's rate sample from the opponent's defensive sample and lets teams cross the boundary independently. Current samples remain small, so the model retains neutral shrinkage and exposes counts. No pace, injury, starting-lineup or home-court adjustment is applied.

`data/game_logs_2026.csv.gz` contains 26,547 player appearances across all **1,230 2025–26 regular-season games**, retrieved 2026-10-19 from ESPN team schedules and summary box scores. Each of 30 teams has 82 games. The NBA Cup Championship is explicitly excluded even though ESPN tags it season type 2. Source details are in `data/game_logs_2026.csv.source.json`. Current approved archetypes/manual overrides label these historical appearances; archived labels cover players outside the current roster. Exact unambiguous canonical/workbook name matching links providers; unmatched players do not contribute cluster samples. Full-season averages derived from these box scores supersede partial archived stat baselines where matched; archived feature data remains unchanged for clustering. RotoWire still covers players without actual prior-season profiles.

Rebuild the historical snapshot with `python matchup_data.py --year 2026 --cache /path/to/local/boxscore-cache`. The cache holds raw source JSON outside the repository. Current-season schedules/box scores refresh every five minutes once a team reaches the matchup boundary; completed box scores are cached locally to avoid downloading them again. Every box score must match the requested season, type and completed status. Partial feed failures produce a visible warning.

### Sportsbook comparison and juice-adjusted totals

The player projection table now includes **Book line**, **Book implied total**, and **Edge %** and **Play** for PTS/REB/AST/3PM/STL/BLK/TOV/PRA. Select FanDuel (default), DraftKings, or BetMGM. Percentage edge equals `(Projected / Book implied total - 1) × 100`, rounded to two decimals. Only edges strictly above +5% receive an Over label and green shading; edges strictly below −5% receive an Under label and red shading. Exactly ±5%, smaller edges, zero/missing implied totals and missing projections remain unmarked. Color intensity increases continuously from pale at 5% to saturated at 25% (clipped thereafter). Only the Edge % and Play cells are highlighted; projected totals are plain. The 5% rule is an extra projection buffer on top of the already no-vig model estimate, not a calculated wagering ROI. This is a numerical projection comparison, not an EV calculation or bet recommendation.

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
numerical core-stat previews** at zero current-season games. Mario Hezonja remains
unavailable: he lacks a 2025–26 NBA profile and is absent from the RotoWire export.
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
player/stat comparisons, ranked by absolute signed percent gap between the
opponent-adjusted projection and the no-vig, Poisson-implied book total. This is
not a probability edge or expected return. Both sides' American prices and the
posted line remain visible. Multiple props for one player can be correlated.

The >5% play filter is on by default; exactly ±5% does not qualify. Disable it to
view the strongest remaining comparisons without marking them as plays. Existing
continuous edge colors apply. If fewer than 20 props qualify, only real eligible
rows appear. CSV downloads include quote timestamps and matchup sample coverage.
Game buttons include future sportsbook events that match the selected date's
ESPN regular-season schedule on home/away teams and tip time (within five
minutes). Past, postponed, canceled, suspended or delayed events are excluded.
There is no fallback to a different game, book, one-sided line, stale quote, or
invented projection. Both paired prices must be current within 15 minutes.

Rankings use automatic rounded minutes (workbook for the first three team games,
season MPG thereafter), the existing game-five stat/matchup policy, and approved
manual archetype overrides. The individual player's minutes slider still affects
that preview only. Zero/unavailable minutes, missing statistical baselines or
incomplete matchup history exclude a player from this table; the coverage
expander explains the underlying inputs. Props are loaded only for the selected
edge game, cached for five minutes, and refreshed by the existing refresh button.
