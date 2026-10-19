"""Mebounds: rebounds a player grabs off their own missed shot.

Definition used here: an offensive rebound (ESPN type_id 156) credited to a
player, where the most recent live event in the same game and period is a
missed shot attempt (field goal or free throw) by that same player. The
lookback skips administrative rows (substitutions, timeouts, replay reviews,
end-of-period markers) but nothing else -- a jumpball or any other live event
between the miss and the board breaks the link.

Data: sportsdataverse ESPN play-by-play parquet dumps (wehoop / hoopR), same
schema for both leagues. Regular season only (season_type == 2), minus the
exhibition games ESPN files under season_type 2 (EXHIBITION_GAME_IDS).

Corrected October 19, 2026 after a cross-feed check against the NBA's own
play-by-play (python/cross_feed_check.py):
  * ESPN's season_type 2 includes each season's All-Star game(s) and the
    in-season cup final (NBA Cup, WNBA Commissioner's Cup). Neither counts in
    official regular-season statistics; both are now excluded.
  * The admin list missed several review and timeout codes ("Challenge",
    "Coach's Challenge (Supported)", "No Timeout", ...), and the lookback
    skipped at most two admin rows, so a challenge sequence between a miss
    and its rebound broke the link. Every admin row is now skipped, as the
    definition always said.
  * Games played counted box-score rows, including did-not-play rows; it now
    counts games the player appeared in.
"""

from pathlib import Path

import polars as pl

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = Path(__file__).resolve().parent.parent / "output"
OUT.mkdir(exist_ok=True)

OREB, DREB = 156, 155
ADMIN_TYPES = (
    "Substitution",
    "Full Timeout",
    "Short Timeout",
    "Official Timeout",
    "No Timeout",          # the WNBA files' generic team timeout
    "Reset Timeout",
    "End Period",
    "End Game",
    "Challenge",
    "Coach's Challenge (Overturned)",
    "Coach's Challenge (Upheld)",
    "Coach's Challenge (Stands)",
    "Coach's Challenge (Supported)",
    "Coach's Challenge (replaycenter)",
    "Ref-Initiated Review (Overturned)",
    "Ref-Initiated Review (Upheld)",
    "Ref-Initiated Review (Stands)",
    "Ref-Initiated Review (Supported)",
    "Not Available",       # blank review marker rows, e.g. "(01:48) [Hornets] "
    "Instant Replay",
)

# ESPN game IDs filed as season_type 2 that are not regular-season games:
# All-Star games and the in-season cup finals. Found by matching every ESPN
# game to the NBA's regular-season game list (python/cross_feed_check.py);
# that check fails if a new unmatched game appears.
EXHIBITION_GAME_IDS = {
    ("nba", 2024): (401607495, 401623259),             # NBA Cup final; All-Star
    ("nba", 2025): (401734908, 401752955, 401752956, 401752957),  # Cup final; All-Star x3
    ("nba", 2026): (401809839, 401838140, 401838141, 401838142, 401838143),
    ("wnba", 2024): (401677672, 401620458),            # Commissioner's Cup final; All-Star
    ("wnba", 2025): (401736430, 401781604),
    ("wnba", 2026): (401857321, 401857320),
}


def mebound_flags(pbp: pl.DataFrame) -> pl.DataFrame:
    """Return offensive-rebound rows flagged as mebounds (or not).

    ESPN's game_play_number runs in game-clock order (checked October 19,
    2026: no row in any of the six files has more time left than the row
    before it in the same period), so it is a safe ordering key."""
    pbp = pbp.sort(["game_id", "game_play_number"])
    live = pbp.filter(~pl.col("type_text").is_in(ADMIN_TYPES)).with_columns(
        prev_miss=(pl.col("shooting_play") & ~pl.col("scoring_play")).shift(1).over("game_id"),
        prev_athlete=pl.col("athlete_id_1").shift(1).over("game_id"),
        prev_period=pl.col("period_number").shift(1).over("game_id"),
        prev_type=pl.col("type_text").shift(1).over("game_id"),
    )
    is_me = (
        pl.col("prev_miss").fill_null(False)
        & (pl.col("prev_period") == pl.col("period_number"))
        & (pl.col("prev_athlete") == pl.col("athlete_id_1"))
    ).fill_null(False)
    orebs = live.filter((pl.col("type_id") == OREB) & pl.col("athlete_id_1").is_not_null())
    return orebs.with_columns(
        is_me.alias("mebound"),
        pl.when(is_me).then(pl.col("prev_type").str.contains("Free Throw")).alias("miss_was_ft"),
    ).drop("prev_miss", "prev_athlete", "prev_period", "prev_type")


def load_pbp(league: str, year: int) -> pl.DataFrame:
    excl = EXHIBITION_GAME_IDS.get((league, year), ())
    return pl.read_parquet(DATA / f"{league}_pbp_{year}.parquet").filter(
        (pl.col("season_type") == 2) & ~pl.col("game_id").is_in(excl))


def load_box(league: str, year: int) -> pl.DataFrame:
    excl = EXHIBITION_GAME_IDS.get((league, year), ())
    return pl.read_parquet(DATA / f"{league}_player_box_{year}.parquet").filter(
        (pl.col("season_type") == 2) & ~pl.col("game_id").is_in(excl))


def season_summary(league: str, year: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per-player rebound/mebound counts and box-score validation frame."""
    pbp = load_pbp(league, year)
    box = load_box(league, year)

    reb_rows = pbp.filter(
        pl.col("type_id").is_in([OREB, DREB]) & pl.col("athlete_id_1").is_not_null()
    )
    reb = reb_rows.group_by("athlete_id_1").agg(
        pl.len().alias("pbp_rebounds"),
        (pl.col("type_id") == OREB).sum().alias("pbp_orebs"),
    )

    me = mebound_flags(pbp).group_by("athlete_id_1").agg(
        pl.col("mebound").sum().alias("mebounds"),
        (pl.col("mebound") & pl.col("miss_was_ft")).sum().alias("mebounds_ft"),
    )

    names = box.sort("game_date", "game_id").group_by("athlete_id").agg(
        pl.col("athlete_display_name").last().alias("player"),
        pl.col("team_abbreviation").last().alias("team"),
        (~pl.col("did_not_play").fill_null(False)).sum().alias("games"),
        pl.col("rebounds").sum().alias("box_rebounds"),
        pl.col("offensive_rebounds").sum().alias("box_orebs"),
    )

    out = (
        names.join(reb, left_on="athlete_id", right_on="athlete_id_1", how="left")
        .join(me, left_on="athlete_id", right_on="athlete_id_1", how="left")
        .with_columns(pl.col("pbp_rebounds", "pbp_orebs", "mebounds", "mebounds_ft").fill_null(0))
        .with_columns(
            (pl.col("mebounds") / pl.col("pbp_rebounds")).round(4).alias("mebound_share_of_reb"),
            (pl.col("mebounds") / pl.col("pbp_orebs")).round(4).alias("mebound_share_of_oreb"),
        )
        .sort(["mebounds", "pbp_rebounds", "athlete_id"], descending=[True, True, False])
    )

    check = out.select(
        (pl.col("pbp_rebounds") - pl.col("box_rebounds")).abs().sum().alias("abs_reb_diff"),
        pl.col("box_rebounds").sum().alias("box_total"),
        pl.col("pbp_rebounds").sum().alias("pbp_total"),
    )
    return out, check


def main() -> None:
    frames = []
    for league, years in (("wnba", (2024, 2025, 2026)), ("nba", (2024, 2025, 2026))):
        for year in years:
            out, check = season_summary(league, year)
            label = f"{year - 1}-{str(year)[2:]}" if league == "nba" else str(year)
            out = out.with_columns(pl.lit(league.upper()).alias("league"), pl.lit(label).alias("season"))
            frames.append(out)
            c = check.row(0, named=True)
            print(
                f"{league.upper()} {label}: pbp rebounds {c['pbp_total']} vs box {c['box_total']} "
                f"(sum abs per-player diff {c['abs_reb_diff']})"
            )

    all_out = pl.concat(frames)
    all_out.write_parquet(OUT / "mebounds_by_player_season.parquet")

    reese = all_out.filter(pl.col("player") == "Angel Reese").select(
        "season", "team", "games", "box_rebounds", "pbp_rebounds", "pbp_orebs",
        "mebounds", "mebounds_ft", "mebound_share_of_reb", "mebound_share_of_oreb",
    )
    reese.write_csv(OUT / "angel_reese_mebounds.csv")
    print("\nAngel Reese by season (regular season):")
    print(reese)

    for league in ("WNBA", "NBA"):
        top = (
            all_out.filter(pl.col("league") == league)
            .sort(["season", "mebounds", "pbp_rebounds", "player"],
                  descending=[False, True, True, False])
            .group_by("season", maintain_order=True)
            .head(10)
            .select(
                "season", "player", "team", "games", "pbp_rebounds", "pbp_orebs",
                "mebounds", "mebound_share_of_reb", "mebound_share_of_oreb",
            )
        )
        top.write_csv(OUT / f"{league.lower()}_top_mebounders.csv")
        print(f"\n{league} top 10 mebounders per season:")
        with pl.Config(tbl_rows=40):
            print(top)


if __name__ == "__main__":
    main()
