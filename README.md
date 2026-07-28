# Mebounds: Angel Reese and the NBA's own-miss rebounders

A study in the [basketball-data-science](https://github.com/ismayc/basketball-data-science)
family. A **"mebound"** — a term coined for Angel Reese — is a rebound a player
grabs off their *own* missed shot. This study measures what share of Reese's
career rebounds have been mebounds, season by season, and finds the top
mebounders in the NBA over the same span.

## Definition

A mebound is an **offensive rebound credited to the same player who missed the
immediately preceding shot attempt** (field goal or free throw) in the same
period. The lookback skips administrative rows (substitutions, timeouts,
replay reviews) but nothing else — a jumpball or any other live event between
the miss and the board breaks the link.

## Data

ESPN play-by-play parquet dumps published by
[sportsdataverse](https://github.com/sportsdataverse) (the data behind the
`wehoop` and `hoopR` packages) — identical schema for both leagues, so one
code path (`python/compute_mebounds.py`) handles WNBA and NBA.

- **Seasons:** WNBA 2024, 2025, 2026 (Reese's career; 2026 through games of
  2026-07-25) and the NBA seasons ending in the same years: 2023-24, 2024-25,
  2025-26.
- **Scope:** regular season only (`season_type == 2`).
- **Validation:** per-player play-by-play rebound totals were reconciled
  against official box scores: WNBA seasons match within 1 rebound
  league-wide; NBA seasons within ~0.03% (a handful of events ESPN logged
  without a player).

## Results

### Angel Reese (regular season)

| Season | Team | G | Reb | OReb | Mebounds | % of all reb | % of OReb |
|---|---|---:|---:|---:|---:|---:|---:|
| 2024 | CHI | 40 | 457 | 177 | 59 | 12.9% | 33.3% |
| 2025 | CHI | 45 | 386 | 128 | 45 | 11.7% | 35.2% |
| 2026* | ATL | 26 | 293 | 126 | 32 | 10.9% | 25.4% |
| **Career** | | **111** | **1,136** | **431** | **136** | **12.0%** | **31.6%** |

\*2026 in progress, through 2026-07-25. All of Reese's 136 mebounds came off
her own missed field goals — zero off her own missed free throws.

The nickname is earned: Reese has led the WNBA in mebound **count** in all
three of her seasons, and among every WNBA player with 200+ rebounds she also
ranks **#1 in mebound share of total rebounds** in all three seasons.

### NBA top mebounders by count (regular season)

| Season | Top 5 (mebounds) |
|---|---|
| 2023-24 | Jokić 95 · Sabonis 61 · Capela 60 · Drummond 57 · Gordon 56 |
| 2024-25 | Jokić 84 · Şengün 79 · Antetokounmpo 56 · Zubac 50 · Sabonis 48 |
| 2025-26 | Jokić 73 · Şengün 68 · Towns 51 · Barnes 49 · Brown/Clingan/Zion/Siakam/Achiuwa 40-41 |

Nikola Jokić is the NBA's mebound king in all three seasons, though at ~9% of
his rebounds — below Reese's share.

### NBA mebound-share leaders (min 300 rebounds)

| Season | Top 3 by share of all rebounds |
|---|---|
| 2023-24 | Zion Williamson 13.1% · Jimmy Butler 12.0% · Aaron Gordon 11.9% |
| 2024-25 | Devin Booker 12.8% · Jaylen Brown 12.7% · Keldon Johnson 11.3% |
| 2025-26 | Zion Williamson 11.3% · Alperen Şengün 10.5% · Tari Eason 10.4% |

Reese's rookie-year 12.9% would rank near the very top of the NBA in any of
these seasons; only Zion Williamson's 13.1% (2023-24) tops it.

## Reproduce

```bash
python python/compute_mebounds.py
```

Outputs land in `output/`: `angel_reese_mebounds.csv`,
`wnba_top_mebounders.csv`, `nba_top_mebounders.csv`, and the full
per-player-season table `mebounds_by_player_season.parquet`.
