"""Cross-feed check: mebounds from ESPN play-by-play vs the NBA's own PlayByPlayV3.

The study's numbers come from ESPN play-by-play (python/compute_mebounds.py).
This script applies the same definition to an independent feed, the NBA's
PlayByPlayV3 rows cached by the sibling playbyplay-study
(../playbyplay-study/data/raw_pbp/<gameId>.parquet), and compares the two
per player-season for every game both feeds cover.

Definition on V3 (mirrors the ESPN one): a player rebound is a mebound when
the most recent live event in the same game and period is a missed shot
(field goal, or free throw whose description starts "MISS") by the same
personId. Skipped as non-live: substitutions, timeouts, instant replays,
period markers, and the blank-actionType BLOCK/STEAL companion rows (ESPN
folds blocks and steals into the shot or turnover row, so V3's separate rows
must be skipped to mirror it). Offensive rebounds per player come from the
running "Off:N" counter in each rebound's description.

Ordering: V3 events are sorted by (period, seconds left descending,
actionNumber), never by actionNumber alone, because the feed appends
late-logged events with a high actionNumber and an early clock (see
playbyplay-study's October 19, 2026 correction). ESPN's game_play_number was
checked to be clock-monotone in every game (0 rows where the clock runs
backwards), so ordering ESPN by it is safe.

Games are matched by Eastern date (from the NBA liveData feed's first
timeActual) and final score; players by co-occurrence of identical events
(same game, period, clock, and event class) in the two feeds.

Run:   ../.venv/bin/python python/cross_feed_check.py           (rebuild + gate)
       ../.venv/bin/python python/cross_feed_check.py --check   (gate only, reads output/)
       ../.venv/bin/python python/cross_feed_check.py --inject order        (V3 in
           actionNumber order, NBA 2023-24; the gate must fail)
       ../.venv/bin/python python/cross_feed_check.py --inject exhibitions  (ESPN
           All-Star and cup-final games kept, WNBA 2024; the gate must fail)

Outputs: output/cross_feed_comparison.csv (one row per league-season),
output/cross_feed_players.csv (player-seasons with 10+ mebounds in either feed
or any difference), output/cross_feed_causes.csv (every disagreement with its
diagnosed cause and the rows behind it).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_mebounds as cm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"
PBP = ROOT.parent / "playbyplay-study" / "data"
RAW = PBP / "raw_pbp"
LIVE = PBP / "pbpstats_live" / "pbp"

# ESPN file year -> (NBA game-ID prefix, display label). NBA's 2023-24 season
# is ESPN's 2024 file (the year the season ends); the WNBA labels agree.
SEASONS = [
    ("nba", 2024, "00223", "2023-24"),
    ("nba", 2025, "00224", "2024-25"),
    ("nba", 2026, "00225", "2025-26"),
    ("wnba", 2024, "10224", "2024"),
    ("wnba", 2025, "10225", "2025"),
    ("wnba", 2026, "10226", "2026"),
]

V3_SKIP = ("", "Substitution", "Timeout", "Instant Replay", "period")

# Gate tolerances, set from the October 19, 2026 diagnosis (README,
# "Cross-feed check"). All 66 disagreements traced there are logging
# differences between the feeds (who was credited, events one feed lacks, two
# tie-clock orderings), none a definition or code difference. Observed worst
# cases: league-season net difference 0.09%, agreement 99.57%, player-season
# |diff| 1. Each tolerance leaves about 2-5x headroom over those so a data
# refresh does not trip it, while the known trap (V3 in actionNumber order)
# and the exhibition-game bug both fail it (--inject).
TOL_LEAGUE_SEASON_PCT = 0.5   # |ESPN - NBA| / NBA mebounds, per league-season
TOL_EVENT_AGREE_PCT = 99.0    # mebounds in both feeds / mebounds in either, per league-season
TOL_PLAYER_ABS = 2            # per player-season |diff| for anyone with 20+ mebounds
TOL_PLAYER_MIN = 20
TOL_UNMATCHED_GAMES = 0       # ESPN games (exhibitions excluded) with no NBA game

# Injection switches (--inject only): reproduce the two failure modes.
V3_ORDER = "clock"            # "action" = sort by actionNumber, the known trap
INCLUDE_EXHIBITIONS = False   # True = the pre-October 19 ESPN game set


def secs_left(clock: pl.Expr) -> pl.Expr:
    """'PT11M40.00S' -> 700.0 seconds left in the period."""
    mins = clock.str.extract(r"PT(\d+)M", 1).cast(pl.Float64)
    secs = clock.str.extract(r"M([\d.]+)S", 1).cast(pl.Float64)
    return mins * 60 + secs


# --------------------------------------------------------------------- NBA V3
def load_v3(prefix: str) -> pl.DataFrame:
    files = sorted(RAW.glob(f"{prefix}*.parquet"))
    if not files:
        raise SystemExit(f"no V3 files for {prefix} in {RAW}")
    frames = [pl.read_parquet(f).with_row_index("file_row") for f in files]
    d = pl.concat(frames, how="diagonal_relaxed")
    d = d.with_columns(
        gameId=pl.col("gameId").str.zfill(10),
        actionNumber=pl.col("actionNumber").cast(pl.Int64),
        period=pl.col("period").cast(pl.Int32),
        secs=secs_left(pl.col("clock")),
        personId=pl.col("personId").cast(pl.Int64),
        teamId=pl.col("teamId").cast(pl.Int64, strict=False),
    )
    # Clock order within the period; actionNumber then file order break ties
    # (a BLOCK row shares its shot's actionNumber and follows it in the file).
    if V3_ORDER == "action":
        return d.sort(["gameId", "actionNumber", "file_row"])
    return d.sort(["gameId", "period", "secs", "actionNumber", "file_row"],
                  descending=[False, False, True, False, False])


def v3_flags(d: pl.DataFrame) -> pl.DataFrame:
    """Every V3 player rebound with its mebound flag and offensive flag."""
    live = d.filter(~pl.col("actionType").is_in(V3_SKIP))
    is_miss = (pl.col("actionType") == "Missed Shot") | (
        (pl.col("actionType") == "Free Throw") & pl.col("description").str.starts_with("MISS"))
    live = live.with_columns(
        is_miss=is_miss,
        is_ft=pl.col("actionType") == "Free Throw",
    ).with_columns(
        prev_miss=pl.col("is_miss").shift(1).over("gameId", "period"),
        prev_ft=pl.col("is_ft").shift(1).over("gameId", "period"),
        prev_person=pl.col("personId").shift(1).over("gameId", "period"),
        prev_type=pl.col("actionType").shift(1).over("gameId", "period"),
        prev_desc=pl.col("description").shift(1).over("gameId", "period"),
    )
    reb = live.filter(
        (pl.col("actionType") == "Rebound") & pl.col("description").str.contains(r"Off:\d+")
    ).with_columns(off_n=pl.col("description").str.extract(r"Off:(\d+)", 1).cast(pl.Int32))
    # Offensive flag from the running counter, read in logging (actionNumber) order.
    reb = reb.sort("gameId", "personId", "actionNumber").with_columns(
        oreb=(pl.col("off_n") - pl.col("off_n").shift(1).over("gameId", "personId").fill_null(0)) > 0
    )
    return reb.with_columns(
        mebound=(pl.col("prev_miss").fill_null(False) & (pl.col("prev_person") == pl.col("personId"))),
        miss_was_ft=pl.col("prev_ft"),
    )


def v3_game_index(prefix: str, d: pl.DataFrame) -> pl.DataFrame:
    """gameId -> Eastern date (liveData's first timeActual) and final score."""
    rows = []
    for f in sorted(LIVE.glob(f"live_{prefix}*.json")):
        acts = json.loads(f.read_text())["game"]["actions"]
        t = next((a["timeActual"] for a in acts if a.get("timeActual")), None)
        rows.append({"gameId": f.stem.removeprefix("live_"), "t": t})
    idx = pl.DataFrame(rows).with_columns(
        date=pl.col("t").str.to_datetime(time_zone="UTC", strict=False)
        .dt.convert_time_zone("America/New_York").dt.date()
    ).drop("t")
    score = d.with_columns(
        h=pl.col("scoreHome").cast(pl.Int32, strict=False),
        a=pl.col("scoreAway").cast(pl.Int32, strict=False),
    ).group_by("gameId").agg(
        home_final=pl.col("h").max(), away_final=pl.col("a").max(),
        home=pl.col("teamTricode").filter(pl.col("location") == "h").mode().first(),
        away=pl.col("teamTricode").filter(pl.col("location") == "v").mode().first(),
    )
    return score.join(idx, on="gameId", how="left")


# ----------------------------------------------------------------------- ESPN
def load_espn(league: str, year: int) -> pl.DataFrame:
    """ESPN rows as the study uses them (exhibitions already excluded)."""
    if INCLUDE_EXHIBITIONS:
        d = pl.read_parquet(DATA / f"{league}_pbp_{year}.parquet").filter(pl.col("season_type") == 2)
    else:
        d = cm.load_pbp(league, year)
    return d.with_columns(secs=pl.col("start_quarter_seconds_remaining"))


def espn_game_index(e: pl.DataFrame) -> pl.DataFrame:
    return e.group_by("game_id").agg(
        date=pl.col("game_date").first(),
        home_final=pl.col("home_score").max(),
        away_final=pl.col("away_score").max(),
        home_espn=pl.col("home_team_abbrev").first(),
        away_espn=pl.col("away_team_abbrev").first(),
    )


def _join_dates(eg: pl.DataFrame, ng: pl.DataFrame, on: list[str]) -> pl.DataFrame:
    cands = []
    for shift in (0, -1, 1):
        cands.append(
            eg.with_columns(date=pl.col("date") + pl.duration(days=shift))
            .join(ng, on=on, how="inner")
            .with_columns(slip=pl.lit(abs(shift)))
        )
    c = pl.concat(cands).sort("slip", "game_id", "gameId", maintain_order=True)
    c = c.unique(["game_id", "gameId"], keep="first", maintain_order=True)
    # Only unambiguous pairs: drop any ESPN game or NBA game with two candidates
    # at its best slip.
    best_e = c.group_by("game_id").agg(pl.col("slip").min().alias("bs"))
    c = c.join(best_e, on="game_id").filter(pl.col("slip") == pl.col("bs")).drop("bs")
    c = c.filter(pl.len().over("game_id") == 1)
    best_n = c.group_by("gameId").agg(pl.col("slip").min().alias("bs"))
    c = c.join(best_n, on="gameId").filter(pl.col("slip") == pl.col("bs")).drop("bs")
    return c.filter(pl.len().over("gameId") == 1)


def match_games(eg: pl.DataFrame, ng: pl.DataFrame) -> pl.DataFrame:
    """ESPN game_id <-> NBA gameId. Pass 1: (date, home final, away final),
    allowing a one-day slip. Pass 1 teaches the ESPN-abbreviation -> NBA-tricode
    map; pass 2 matches the rest on (date, home team, away team), which catches
    games whose final score differs between the feeds. One-to-one only."""
    p1 = _join_dates(eg, ng, ["date", "home_final", "away_final"])
    tri = pl.concat([
        p1.select(pl.col("home_espn").alias("espn"), pl.col("home").alias("tri")),
        p1.select(pl.col("away_espn").alias("espn"), pl.col("away").alias("tri")),
    ]).group_by("espn").agg(pl.col("tri").mode().first())
    rest_e = eg.join(p1, on="game_id", how="anti").join(
        tri.rename({"espn": "home_espn", "tri": "home"}), on="home_espn").join(
        tri.rename({"espn": "away_espn", "tri": "away"}), on="away_espn")
    rest_n = ng.join(p1, on="gameId", how="anti").drop("home_final", "away_final")
    p2 = _join_dates(rest_e.drop("home_final", "away_final"), rest_n, ["date", "home", "away"])
    return pl.concat([p1.select("game_id", "gameId"), p2.select("game_id", "gameId")])


def event_keys_espn(e: pl.DataFrame) -> pl.DataFrame:
    cls = (
        pl.when(pl.col("type_id").is_in([cm.OREB, cm.DREB])).then(pl.lit("reb"))
        .when(pl.col("shooting_play") & pl.col("scoring_play")).then(pl.lit("make"))
        .when(pl.col("shooting_play")).then(pl.lit("miss"))
        .otherwise(None)
    )
    return e.filter(pl.col("athlete_id_1").is_not_null()).with_columns(cls=cls).filter(
        pl.col("cls").is_not_null()
    ).select("game_id", pl.col("period_number").alias("period"),
             pl.col("secs").round(1).alias("s"), "cls", "athlete_id_1")


def event_keys_v3(d: pl.DataFrame) -> pl.DataFrame:
    miss_ft = (pl.col("actionType") == "Free Throw") & pl.col("description").str.starts_with("MISS")
    cls = (
        pl.when((pl.col("actionType") == "Rebound") & pl.col("description").str.contains("Off:"))
        .then(pl.lit("reb"))
        .when((pl.col("actionType") == "Missed Shot") | miss_ft).then(pl.lit("miss"))
        .when(pl.col("actionType").is_in(["Made Shot", "Free Throw"])).then(pl.lit("make"))
        .otherwise(None)
    )
    return d.with_columns(cls=cls).filter(pl.col("cls").is_not_null()).select(
        "gameId", "period", pl.col("secs").round(1).alias("s"), "cls", "personId",
        "playerName", "playerNameI")


def match_players(e: pl.DataFrame, d: pl.DataFrame, games: pl.DataFrame,
                  league: str, year: int) -> pl.DataFrame:
    """ESPN athlete_id -> NBA personId by the most co-occurring identical events."""
    ek = event_keys_espn(e).join(games, on="game_id")
    nk = event_keys_v3(d)
    pairs = ek.join(nk, on=["gameId", "period", "s", "cls"])
    counts = pairs.group_by("athlete_id_1", "personId").agg(n=pl.len())
    tot = ek.group_by("athlete_id_1").agg(n_events=pl.len())
    best = (counts.sort("n", descending=True).group_by("athlete_id_1").agg(
        personId=pl.col("personId").first(), n=pl.col("n").first(),
        n2=pl.col("n").slice(1, 1).first()))
    best = best.join(tot, on="athlete_id_1").with_columns(
        share=pl.col("n") / pl.col("n_events"))
    # Accept when the best partner co-occurs on a majority of the player's
    # events and clearly beats the runner-up.
    ok = (pl.col("share") >= 0.5) & ((pl.col("n2").is_null()) | (pl.col("n") >= 3 * pl.col("n2")))
    best = best.filter(ok)
    # One personId per ESPN id; if two ESPN ids claim one personId keep the stronger.
    by_events = best.sort("n", descending=True).unique("personId", keep="first").select(
        "athlete_id_1", "personId")
    # Fallback by name ("F. Last", accents folded) for players the events left
    # unmapped (a handful with very few events), only when the name is unique.
    ename = (e.filter(pl.col("athlete_id_1").is_not_null()).select("athlete_id_1").unique()
             .join(by_events, on="athlete_id_1", how="anti")
             .join(box_names(league, year), on="athlete_id_1", how="left"))
    nname = (nk.select("personId", "playerNameI").unique("personId")
             .join(by_events, on="personId", how="anti")
             .with_columns(key=fold(pl.col("playerNameI"))))
    ename = ename.filter(pl.col("key").is_not_null())
    ename = ename.filter(pl.len().over("key") == 1)
    nname = nname.filter(pl.len().over("key") == 1)
    by_name = ename.join(nname, on="key").select("athlete_id_1", "personId")
    return pl.concat([by_events, by_name])


def fold(expr: pl.Expr) -> pl.Expr:
    """'N. Jokić' / 'Nikola Jokic' -> 'n. jokic': accents dropped, lowercased."""
    return (expr.str.normalize("NFKD").str.replace_all(r"[^\x00-\x7f]", "")
            .str.to_lowercase().str.strip_chars())


def box_names(league: str, year: int) -> pl.DataFrame:
    """ESPN athlete_id -> 'f. last' name key, from the season's box scores."""
    b = (pl.read_parquet(DATA / f"{league}_player_box_{year}.parquet",
                         columns=["athlete_id", "athlete_display_name"])
         .unique("athlete_id"))
    return b.select(
        pl.col("athlete_id").alias("athlete_id_1"),
        key=fold(pl.col("athlete_display_name").str.extract(r"^(\S)", 1) + ". "
                 + pl.col("athlete_display_name").str.extract(r"^\S+\s+(.*)$", 1)),
    )


# ------------------------------------------------------------------- compare
def compare_season(league: str, year: int, prefix: str, label: str) -> dict:
    e = load_espn(league, year)
    d = load_v3(prefix)
    games = match_games(espn_game_index(e), v3_game_index(prefix, d))
    unmatched_espn_games = e.select("game_id").unique().join(games, on="game_id", how="anti")
    e = e.join(games, on="game_id", how="semi")
    d = d.join(games, on="gameId", how="semi")

    ef = cm.mebound_flags(e)                       # ESPN orebs + flag, as published
    nf = v3_flags(d)
    pmap = match_players(e, d, games, league, year)

    box = pl.read_parquet(DATA / f"{league}_player_box_{year}.parquet").filter(
        pl.col("season_type") == 2).group_by("athlete_id").agg(
        player=pl.col("athlete_display_name").last())

    ep = ef.group_by("athlete_id_1").agg(
        espn_orebs=pl.len(), espn_mebounds=pl.col("mebound").sum())
    np_ = nf.group_by("personId").agg(
        nba_orebs=pl.col("oreb").sum(), nba_mebounds=pl.col("mebound").sum(),
        nba_name=pl.col("playerNameI").last())
    np_ = np_.join(pmap, on="personId", how="left")
    both = (ep.join(np_, on="athlete_id_1", how="full", coalesce=True)
            .join(box, left_on="athlete_id_1", right_on="athlete_id", how="left")
            .with_columns(league=pl.lit(league.upper()), season=pl.lit(label)))

    # Event-level comparison, one-to-one by construction: mebound counts per
    # (game, period, player) in each feed. A cell where the counts differ is a
    # disagreement; the diagnosis reads those cells' sequences in both feeds.
    em = (ef.filter(pl.col("mebound")).join(pmap, on="athlete_id_1", how="left")
          .join(games, on="game_id")
          .group_by("gameId", pl.col("period_number").alias("period"), "personId", "athlete_id_1")
          .agg(espn=pl.len()))
    nm = (nf.filter(pl.col("mebound")).join(pmap, on="personId", how="left")
          .group_by("gameId", "period", "personId", "athlete_id_1").agg(nba=pl.len()))
    cells = em.join(nm, on=["gameId", "period", "personId", "athlete_id_1"],
                    how="full", coalesce=True, nulls_equal=True).with_columns(
        pl.col("espn", "nba").fill_null(0).cast(pl.Int64))
    return {
        "both": both, "cells": cells, "espn_flags": ef, "nba_flags": nf, "pmap": pmap,
        "games": games, "e": e, "d": d,
        "unmatched_espn_games": unmatched_espn_games,
        "n_games": games.height,
    }


def summarize(r: dict, league: str, label: str) -> dict:
    b, c = r["both"], r["cells"]
    agree = int(c.select(pl.min_horizontal("espn", "nba")).sum().item())
    union = int(c.select(pl.max_horizontal("espn", "nba")).sum().item())
    return {
        "league": league.upper(), "season": label, "games_compared": r["n_games"],
        "espn_games_unmatched": r["unmatched_espn_games"].height,
        "espn_orebs": int(r["espn_flags"].height),
        "nba_orebs": int(r["nba_flags"]["oreb"].sum()),
        "espn_mebounds": int(r["espn_flags"]["mebound"].sum()),
        "nba_mebounds": int(r["nba_flags"]["mebound"].sum()),
        "mebounds_in_both": agree,
        "espn_only": int((c["espn"] - c["nba"]).clip(lower_bound=0).sum()),
        "nba_only": int((c["nba"] - c["espn"]).clip(lower_bound=0).sum()),
        "agreement_pct": round(100 * agree / union, 3) if union else None,
        "players_espn_unmapped": int(b.filter(pl.col("personId").is_null()
                                              & (pl.col("espn_orebs").fill_null(0) > 0)).height),
        "players_nba_unmapped": int(b.filter(pl.col("athlete_id_1").is_null()
                                             & (pl.col("nba_orebs").fill_null(0) > 0)).height),
    }


# ----------------------------------------------------------------- diagnose
def _espn_live(e: pl.DataFrame) -> pl.DataFrame:
    cls = (
        pl.when(pl.col("type_id").is_in([cm.OREB, cm.DREB]) & pl.col("athlete_id_1").is_null())
        .then(pl.lit("teamreb"))
        .when(pl.col("type_id").is_in([cm.OREB, cm.DREB])).then(pl.lit("reb"))
        .when(pl.col("shooting_play") & pl.col("scoring_play")).then(pl.lit("make"))
        .when(pl.col("shooting_play")).then(pl.lit("miss"))
        .otherwise(pl.lit("other"))
    )
    return (e.sort("game_id", "game_play_number")
            .filter(~pl.col("type_text").is_in(cm.ADMIN_TYPES))
            .with_columns(cls=cls)
            .select("gameId", pl.col("period_number").alias("period"), "secs", "cls",
                    pl.col("athlete_id_1").alias("who"), pl.col("game_play_number").alias("n"),
                    pl.col("text").alias("desc")))


def _v3_live(d: pl.DataFrame, pmap: pl.DataFrame) -> pl.DataFrame:
    miss_ft = (pl.col("actionType") == "Free Throw") & pl.col("description").str.starts_with("MISS")
    cls = (
        pl.when((pl.col("actionType") == "Rebound") & pl.col("description").str.contains("Off:"))
        .then(pl.lit("reb"))
        .when(pl.col("actionType") == "Rebound").then(pl.lit("teamreb"))
        .when((pl.col("actionType") == "Missed Shot") | miss_ft).then(pl.lit("miss"))
        .when(pl.col("actionType").is_in(["Made Shot", "Free Throw"])).then(pl.lit("make"))
        .otherwise(pl.lit("other"))
    )
    # Actors expressed in ESPN ids so both feeds compare on one key.
    return (d.filter(~pl.col("actionType").is_in(V3_SKIP)).with_columns(cls=cls)
            .join(pmap, on="personId", how="left")
            .select("gameId", "period", "secs", "cls", pl.col("athlete_id_1").alias("who"),
                    pl.col("actionNumber").alias("n"), pl.col("description").alias("desc")))


def _near(f: pl.DataFrame, t: float, tol: float, **eq) -> pl.DataFrame:
    out = f.filter((pl.col("secs") - t).abs() <= tol)
    for k, v in eq.items():
        out = out.filter(pl.col(k) == v) if v is not None else out.filter(pl.col(k).is_null())
    return out


def _fmt(rows: pl.DataFrame) -> str:
    return " | ".join(f"{r['n']} {r['secs']:g}s {r['desc']}" for r in rows.iter_rows(named=True))


def _classify(F: pl.DataFrame, G: pl.DataFrame, t: float, who: int, fname: str, gname: str
              ) -> tuple[str, str]:
    """A mebound at clock t flagged in feed F but not in G: why?

    The player's rebounds within 2 s of t are lined up in each feed's order;
    the flagged one's counterpart in G is the rebound at the same position."""
    f_i = F.with_row_index("i")
    f_i = f_i.with_columns(pm=pl.col("cls").shift(1) == "miss", pw=pl.col("who").shift(1))
    me = f_i.filter((pl.col("secs") == t) & (pl.col("cls") == "reb") & (pl.col("who") == who)
                    & pl.col("pm") & (pl.col("pw") == who))
    i = me["i"][-1]   # with duplicates at one clock, the later copy is the extra one
    miss = f_i.filter(pl.col("i") == i - 1)
    tm = miss["secs"][0]
    f_near = _near(f_i, t, 2.0, cls="reb", who=who)["i"].to_list()
    k = f_near.index(i)
    g_i = G.with_row_index("i")
    g_near = _near(g_i, t, 2.0, cls="reb", who=who)
    if g_near.height <= k:
        other = (_near(g_i, t, 1.5).filter(pl.col("cls").is_in(["reb", "teamreb"]))
                 .join(g_near.select("i"), on="i", how="anti"))
        # A G rebound that F also has (same credit, same clock) is not the
        # counterpart: keep a credit only where G has more of it than F.
        keep = []
        for r in other.iter_rows(named=True):
            n_g = _near(g_i, t, 1.5, cls=r["cls"], who=r["who"]).height
            n_f = _near(F, t, 1.5, cls=r["cls"], who=r["who"]).height
            keep.append(n_g > n_f)
        other = other.filter(pl.Series(keep, dtype=pl.Boolean)) if other.height else other
        if other.filter(pl.col("cls") == "reb").height:
            return "attribution: other player", f"{gname} credits another player: {_fmt(other)}"
        if other.height:
            return "attribution: team rebound", f"{gname} credits the team: {_fmt(other)}"
        return "missing event", (f"{fname} has a rebound by the player at {t:g}s "
                                 f"[{_fmt(f_i.filter(pl.col('i') == i))}] that {gname} lacks")
    j = g_near["i"][k]
    prev_g = g_i.filter(pl.col("i") == j - 1)
    g_miss = _near(G, tm, 1.0, cls="miss")
    if g_miss.height and not g_miss.filter(pl.col("who") == who).height:
        return "attribution: shooter", (f"{fname} credits the miss to the rebounder "
                                        f"[{_fmt(miss)}]; {gname} to another player [{_fmt(g_miss)}]")
    if _near(G, tm, 3.0, cls="miss", who=who).height == 0:
        return "missing event", f"{gname} lacks the miss [{_fmt(miss)}]"
    if prev_g.height == 0:
        return "other", "no previous event in " + gname
    pg = prev_g.row(0, named=True)
    if pg["cls"] in ("reb", "teamreb", "miss", "make"):
        twin = _near(F, pg["secs"], 2.0, cls=pg["cls"], who=pg["who"])
    else:
        twin = _near(F, pg["secs"], 2.0, cls="other")
    if twin.height:
        return "ordering", (f"{gname} puts [{_fmt(prev_g)}] between the miss and the rebound; "
                            f"{fname} has the same event in another position")
    return "missing event", f"{gname} has [{_fmt(prev_g)}] between the miss and the rebound; {fname} lacks it"


def diagnose(r: dict, league: str, label: str) -> pl.DataFrame:
    """One row per mebound flagged in one feed but not the other, with a cause."""
    c = r["cells"].filter(pl.col("espn") != pl.col("nba"))
    games = r["games"]
    e = r["e"].join(games, on="game_id")
    E = _espn_live(e.filter(pl.col("gameId").is_in(c["gameId"].unique().implode())))
    N = _v3_live(r["d"].filter(pl.col("gameId").is_in(c["gameId"].unique().implode())), r["pmap"])
    ef = (r["espn_flags"].filter(pl.col("mebound")).join(games, on="game_id")
          .select("gameId", pl.col("period_number").alias("period"), "secs", "athlete_id_1"))
    nf = (r["nba_flags"].filter(pl.col("mebound")).join(r["pmap"], on="personId", how="left")
          .select("gameId", "period", "secs", "athlete_id_1", "personId"))
    rows = []
    for cell in c.iter_rows(named=True):
        g, per, who = cell["gameId"], cell["period"], cell["athlete_id_1"]
        Eg = E.filter((pl.col("gameId") == g) & (pl.col("period") == per))
        Ng = N.filter((pl.col("gameId") == g) & (pl.col("period") == per))
        if who is None:
            rows.append({"cause": "player unmapped", "flagged_in": "ESPN" if cell["espn"] else "NBA",
                         "gameId": g, "period": per, "secs": None, "athlete_id_1": None,
                         "detail": f"personId {cell['personId']} has no ESPN id"})
            continue
        te = sorted(ef.filter((pl.col("gameId") == g) & (pl.col("period") == per)
                              & (pl.col("athlete_id_1") == who))["secs"].to_list(), reverse=True)
        tn = sorted(nf.filter((pl.col("gameId") == g) & (pl.col("period") == per)
                              & (pl.col("athlete_id_1") == who))["secs"].to_list(), reverse=True)
        # pair mebounds across feeds, closest clocks first (within 2 s)
        pairs = sorted(((abs(x - y), a, b) for a, x in enumerate(te) for b, y in enumerate(tn)
                        if abs(x - y) <= 2.0))
        used_e, used_n = set(), set()
        for _, a, b in pairs:
            if a not in used_e and b not in used_n:
                used_e.add(a)
                used_n.add(b)
        un_e = [x for a, x in enumerate(te) if a not in used_e]
        un_n = [y for b, y in enumerate(tn) if b not in used_n]
        for fl, ts, F, G, fn, gn in (("ESPN", un_e, Eg, Ng, "ESPN", "NBA"),
                                     ("NBA", un_n, Ng, Eg, "NBA", "ESPN")):
            for t in ts:
                cause, detail = _classify(F, G, t, who, fn, gn)
                rows.append({"cause": cause, "flagged_in": fl, "gameId": g, "period": per,
                             "secs": t, "athlete_id_1": who, "detail": detail})
    out = pl.DataFrame(rows, schema={"cause": pl.String, "flagged_in": pl.String,
                                     "gameId": pl.String, "period": pl.Int32, "secs": pl.Float64,
                                     "athlete_id_1": pl.Int64, "detail": pl.String})
    names = r["both"].select(pl.col("athlete_id_1").cast(pl.Int64), "player").unique("athlete_id_1")
    return out.join(names, on="athlete_id_1", how="left").with_columns(
        league=pl.lit(league.upper()), season=pl.lit(label)).select(
        "league", "season", "cause", "flagged_in", "player", "gameId", "period", "secs", "detail")


TABLE = OUT / "cross_feed_comparison.csv"
PLAYERS = OUT / "cross_feed_players.csv"
CAUSES = OUT / "cross_feed_causes.csv"


def tables(seasons=SEASONS, with_causes: bool = True):
    rows, players, causes = [], [], []
    for league, year, prefix, label in seasons:
        r = compare_season(league, year, prefix, label)
        s = summarize(r, league, label)
        print(s)
        rows.append(s)
        if with_causes:
            causes.append(diagnose(r, league, label))
        players.append(r["both"].filter(
            (pl.col("espn_mebounds").fill_null(0) >= 10) | (pl.col("nba_mebounds").fill_null(0) >= 10)
            | (pl.col("espn_mebounds").fill_null(0) != pl.col("nba_mebounds").fill_null(0))
        ).select("league", "season", "player", "nba_name", "athlete_id_1", "personId",
                 "espn_orebs", "nba_orebs", "espn_mebounds", "nba_mebounds"))
    t = pl.DataFrame(rows).with_columns(
        mebound_diff=pl.col("espn_mebounds") - pl.col("nba_mebounds"),
        mebound_diff_pct=(100 * (pl.col("espn_mebounds") - pl.col("nba_mebounds"))
                          / pl.col("nba_mebounds")).round(3),
    )
    p = pl.concat(players).with_columns(
        pl.col("espn_orebs", "nba_orebs", "espn_mebounds", "nba_mebounds").cast(pl.Int64)
    ).with_columns(
        diff=pl.col("espn_mebounds").fill_null(0) - pl.col("nba_mebounds").fill_null(0)
    ).sort(["league", "season", "espn_mebounds", "player"],
           descending=[False, False, True, False], nulls_last=True)
    c = (pl.concat(causes).sort("league", "season", "cause", "gameId", "secs", "flagged_in",
                                    "player") if causes else None)
    return t, p, c


def build() -> None:
    t, p, c = tables()
    t.write_csv(TABLE)
    p.write_csv(PLAYERS)
    c.write_csv(CAUSES)
    with pl.Config(tbl_rows=40):
        print(c.group_by("league", "cause").len().sort("league", "cause"))


def gate(t: pl.DataFrame, p: pl.DataFrame) -> list[str]:
    fails = []
    for row in t.iter_rows(named=True):
        tag = f"{row['league']} {row['season']}"
        if abs(row["mebound_diff_pct"]) > TOL_LEAGUE_SEASON_PCT:
            fails.append(f"{tag}: mebounds differ {row['mebound_diff_pct']}% "
                         f"(ESPN {row['espn_mebounds']} vs NBA {row['nba_mebounds']}) > {TOL_LEAGUE_SEASON_PCT}%")
        if row["agreement_pct"] is None or row["agreement_pct"] < TOL_EVENT_AGREE_PCT:
            fails.append(f"{tag}: event agreement {row['agreement_pct']}% < {TOL_EVENT_AGREE_PCT}%")
        if row["espn_games_unmatched"] > TOL_UNMATCHED_GAMES:
            fails.append(f"{tag}: {row['espn_games_unmatched']} ESPN games have no NBA match")
    big = p.filter(
        (pl.max_horizontal(pl.col("espn_mebounds").fill_null(0), pl.col("nba_mebounds").fill_null(0)) >= TOL_PLAYER_MIN)
        & (pl.col("diff").abs() > TOL_PLAYER_ABS))
    for row in big.iter_rows(named=True):
        fails.append(f"{row['league']} {row['season']} {row['player']}: ESPN {row['espn_mebounds']} "
                     f"vs NBA {row['nba_mebounds']} (|diff| > {TOL_PLAYER_ABS})")
    return fails


def main() -> int:
    global V3_ORDER, INCLUDE_EXHIBITIONS
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="gate the committed tables only")
    ap.add_argument("--inject", choices=["order", "exhibitions"],
                    help="recompute with a known failure mode; the gate must FAIL")
    a = ap.parse_args()
    if a.inject:
        # Nothing is written: recompute in memory with the fault and gate it.
        if a.inject == "order":
            V3_ORDER = "action"
            seasons = [s for s in SEASONS if s[3] == "2023-24"]
        else:
            INCLUDE_EXHIBITIONS = True
            seasons = [s for s in SEASONS if s[0] == "wnba" and s[1] == 2024]
        t, p, _ = tables(seasons, with_causes=False)
        print(f"injected fault: {a.inject}")
    else:
        if not a.check:
            build()
        t, p = pl.read_csv(TABLE), pl.read_csv(PLAYERS)
    fails = gate(t, p)
    if fails:
        print("CROSS-FEED GATE FAILED")
        for f in fails:
            print("  " + f)
        return 1
    print(f"CROSS-FEED GATE PASSED: {t.height} league-seasons, "
          f"{int(t['nba_mebounds'].sum())} NBA-feed mebounds, {p.height} player rows checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
