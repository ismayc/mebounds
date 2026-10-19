# Mebounds: Angel Reese and the NBA's own-miss rebounders

A study in the [basketball-data-science](https://github.com/ismayc/basketball-data-science)
family. A **"mebound"** (a term coined for Angel Reese) is a rebound a player
grabs off their *own* missed shot. This study measures what share of Reese's
career rebounds have been mebounds, season by season, and finds the top
mebounders in the NBA over the same span.

> **Correction, October 19, 2026.** A cross-feed check against the NBA's own
> play-by-play found that ESPN files each season's All-Star games and the
> in-season cup finals (NBA Cup, WNBA Commissioner's Cup) as regular-season
> games. The July 28 numbers counted those 17 exhibition games, and the games
> column counted did-not-play rows. Both are fixed, along with several review and
> timeout codes the lookback did not skip (net effect: 3 NBA mebounds). Reese's
> published line changes from 136 career mebounds on 1,136 rebounds (12.0%) to
> **132 on 1,111 (11.9%)**; 2024 is now 446 rebounds in 34 games, not 457 in 40.
> She still leads the WNBA in mebound count every season, but the claim that
> she ranks first in mebound share every season no longer holds: in 2025
> Elizabeth Williams (11.2%) edges her (11.1%), and both feeds agree on both
> players. The NBA top-five lists keep their order (Jokić 95 to 94 in 2023-24,
> Antetokounmpo 56 to 54 in 2024-25, Towns 51 to 50 in 2025-26); in 2024-25
> Jaylen Brown (12.8%) now edges Devin Booker for the share lead. Tables below
> are the corrected ones. Published before: `git show a32f490:README.md`.

## Definition

A mebound is an **offensive rebound credited to the same player who missed the
immediately preceding shot attempt** (field goal or free throw) in the same
period. The lookback skips administrative rows (substitutions, timeouts,
replay reviews) but nothing else: a jumpball or any other live event between
the miss and the board breaks the link.

## Data

ESPN play-by-play parquet dumps published by
[sportsdataverse](https://github.com/sportsdataverse) (the data behind the
`wehoop` and `hoopR` packages). The schema is identical for both leagues, so one
code path (`python/compute_mebounds.py`) handles WNBA and NBA.

- **Seasons:** WNBA 2024, 2025, 2026 (Reese's career; 2026 through games of
  2026-07-25) and the NBA seasons ending in the same years: 2023-24, 2024-25,
  2025-26.
- **Scope:** regular season only (`season_type == 2`), minus the 17 All-Star
  and cup-final games ESPN files under that type (`EXHIBITION_GAME_IDS`).
  ESPN's NBA 2025-26 file also lacks one regular-season game (CHI vs LAC,
  January 20, 2026).
- **Validation:** per-player play-by-play rebound totals were reconciled
  against ESPN's box scores: WNBA seasons match within 1 rebound league-wide;
  NBA seasons within ~0.02% (a handful of events ESPN logged without a
  player). Mebounds are also checked against a second feed (below).

## Results

### Angel Reese (regular season)

| Season | Team | G | Reb | OReb | Mebounds | % of all reb | % of OReb |
|---|---|---:|---:|---:|---:|---:|---:|
| 2024 | CHI | 34 | 446 | 172 | 58 | 13.0% | 33.7% |
| 2025 | CHI | 30 | 377 | 123 | 42 | 11.1% | 34.1% |
| 2026* | ATL | 25 | 288 | 126 | 32 | 11.1% | 25.4% |
| **Career** | | **89** | **1,111** | **421** | **132** | **11.9%** | **31.4%** |

\*2026 in progress, through 2026-07-25. All of Reese's 132 mebounds came off
her own missed field goals, none off her own missed free throws.

The nickname is earned: Reese has led the WNBA in mebound **count** in all
three of her seasons (58 to Aliyah Boston's 29 in 2024, 42 to Kamilla
Cardoso's 37 in 2025, 32 to 18 in 2026). Among WNBA players with 200+
rebounds she ranks **#1 in mebound share of total rebounds** in 2024 and 2026,
and second in 2025, by 0.07 points behind Elizabeth Williams (25 mebounds on
223 rebounds).

### NBA top mebounders by count (regular season)

| Season | Top 5 (mebounds) |
|---|---|
| 2023-24 | Jokić 94 · Sabonis 61 · Capela 60 · Drummond 57 · Gordon 56 |
| 2024-25 | Jokić 84 · Şengün 79 · Antetokounmpo 54 · Zubac 50 · Sabonis 48 |
| 2025-26 | Jokić 73 · Şengün 68 · Towns 50 · Barnes 49 · Clingan/Brown/Thompson/Achiuwa/Siakam/Zion 40-41 |

Nikola Jokić is the NBA's mebound king in all three seasons, though at ~9% of
his rebounds, below Reese's share.

### NBA mebound-share leaders (min 300 rebounds)

| Season | Top 3 by share of all rebounds |
|---|---|
| 2023-24 | Zion Williamson 13.1% · Jimmy Butler 12.0% · Aaron Gordon 11.9% |
| 2024-25 | Jaylen Brown 12.8% · Devin Booker 12.8% · Keldon Johnson 11.3% |
| 2025-26 | Zion Williamson 11.3% · Alperen Şengün 10.6% · Tari Eason 10.4% |

Reese's rookie-year 13.0% would rank near the very top of the NBA in any of
these seasons; only Zion Williamson's 13.05% (2023-24) tops it.

## Cross-feed check (October 19, 2026)

`python/cross_feed_check.py` applies the same definition to the NBA's own
`PlayByPlayV3` rows, cached by the sibling
[playbyplay-study](https://github.com/ismayc/playbyplay-study), and compares
the two feeds game by game. Games are matched on date and final score (then
date and teams); players by the events both feeds log at the same clock,
with a name fallback (every player with an offensive rebound was mapped).
V3 rows are ordered by game clock within the period, never by `actionNumber`,
because V3 appends late-logged events with high numbers. ESPN's
`game_play_number` was checked and runs in clock order in every game, so the
ESPN script has no ordering assumption of that kind.

| League | Season | Games | ESPN mebounds | NBA mebounds | In both | Agreement |
|---|---|---:|---:|---:|---:|---:|
| NBA | 2023-24 | 1,230 | 5,583 | 5,578 | 5,570 | 99.62% |
| NBA | 2024-25 | 1,230 | 5,583 | 5,579 | 5,570 | 99.61% |
| NBA | 2025-26 | 1,229 | 5,569 | 5,570 | 5,559 | 99.62% |
| WNBA | 2024 | 240 | 973 | 973 | 973 | 100.00% |
| WNBA | 2025 | 286 | 1,119 | 1,119 | 1,119 | 100.00% |
| WNBA | 2026 | 202 | 756 | 756 | 755 | 99.74% |

Agreement is mebounds in both feeds over mebounds in either. All 66
disagreements were read in both feeds and classified
(`output/cross_feed_causes.csv`): 28 are events one feed has and the other
lacks (often V3's late-inserted putback pairs), 20 credit the rebound to a
different player, 10 to the team, 5 credit the miss to a different shooter,
and 3 are a different order of the same events at one clock. None is a
definition or code difference. No player-season differs by more than one
mebound, and Reese matches exactly in every season. The gate fails when a
league-season differs by more than 0.5%, agreement drops below 99%, a
20-mebound player differs by more than 2, or an ESPN game has no NBA match;
`--inject order` (V3 in `actionNumber` order) and `--inject exhibitions` both
fail it.

## Reproduce

```bash
../.venv/bin/python python/compute_mebounds.py
../.venv/bin/python python/build_story_data.py   # story data + page chart data
./run_checks.sh          # rebuilds the cross-feed tables, runs the gate, checks the page
```

Outputs land in `output/`: `angel_reese_mebounds.csv`,
`wnba_top_mebounders.csv`, `nba_top_mebounders.csv`, the full
per-player-season table `mebounds_by_player_season.parquet`, and the
cross-feed tables `cross_feed_comparison.csv`, `cross_feed_players.csv`, and
`cross_feed_causes.csv`. `python/build_story_data.py` writes
`output/story_data.json` (the story page's tables plus every number its prose
quotes), the chart data embedded in `story/index.html`, and the
`docs/index.html` copy that Pages serves. Its `--check` mode, run by
`run_checks.sh`, fails if the JSON does not regenerate byte-identically, if a
number tagged `data-fact` on the page disagrees with the outputs, if a listed
claim no longer holds, or if an untagged number appears in the prose.
