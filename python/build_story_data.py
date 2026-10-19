"""Build the story page's data from the study outputs, and check the page.

    ../.venv/bin/python python/build_story_data.py            write
    ../.venv/bin/python python/build_story_data.py --check    verify, write nothing
    ../.venv/bin/python python/build_story_data.py --check --inject prose

Writing does three things, deterministically:
  1. output/story_data.json: the analysis tables behind the page (the same
     six lists as the July 28 file, rebuilt from
     output/mebounds_by_player_season.parquet) plus a "facts" block holding
     every number the page's prose quotes, formatted as the page prints it.
  2. The `const DATA = ...;` line in story/index.html: the chart data, derived
     from (1) (top fives, display names, scatter highlights, archetype rows).
  3. docs/index.html: a byte-identical copy of story/index.html (Pages serves
     docs/; story/ is the source).

Sources. Everything comes from committed outputs except three counts that
no output holds: play-by-play events, all rebound rows (team rebounds
included), and missed field goals per player. Those are counted from the
local ESPN caches in data/ through compute_mebounds.load_pbp, so the
exhibition filter is the study's own. Game counts come from
output/cross_feed_comparison.csv and are cross-checked against data/.

--check fails (exit 1) when any of these disagree:
  * story_data.json does not regenerate byte-identically;
  * the page's DATA line differs from the regenerated one;
  * docs/index.html differs from story/index.html;
  * an element tagged data-fact="key" does not print facts[key];
  * a sentence listed in CLAIMS is missing from the page, or its
    predicate is false on the outputs;
  * a number in the page's prose is neither tagged nor allowed by
    NUMBER_TOKENS / NUMBER_CONTEXTS
    (the dated correction note, which quotes the July 28 numbers on
    purpose, is exempt from this one scan but not from the fact check).
--inject json|prose|untagged|claim corrupts the in-memory copy first, so
run_checks.sh can prove each failure path fires.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_mebounds as cm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
STORY = ROOT / "story" / "index.html"
DOCS = ROOT / "docs" / "index.html"
JSON_OUT = OUT / "story_data.json"

REESE = "Angel Reese"
SEASONS = {"WNBA": ("2024", "2025", "2026"), "NBA": ("2023-24", "2024-25", "2025-26")}
NBA_FOR = dict(zip(SEASONS["WNBA"], SEASONS["NBA"]))  # same calendar year

# ESPN spells these without diacritics; the page prints them properly.
DISPLAY_NAMES = {"Nikola Jokic": "Nikola Jokić", "Alperen Sengun": "Alperen Şengün",
                 "Luka Doncic": "Luka Dončić"}

# Scatter highlights (blue dots) and their labels, by (ESPN name, season).
SCATTER_HL = {
    ("Nikola Jokic", "2023-24"): {},
    ("Zion Williamson", "2023-24"): {"lab": "zion", "dx": 0, "dy": -10, "anch": "middle"},
    ("Nikola Jokic", "2024-25"): {"lab": "Jokić — led NBA ×3", "dx": 9, "dy": 4},
    ("Alperen Sengun", "2024-25"): {"lab": "Şengün", "dx": 2, "dy": 19, "anch": "middle"},
    ("Jaylen Brown", "2024-25"): {"lab": "Brown", "dx": 10, "dy": 6},
    ("Devin Booker", "2024-25"): {"lab": "Booker", "dx": -10, "dy": 4, "anch": "end"},
    ("Nikola Jokic", "2025-26"): {},
    ("Zion Williamson", "2025-26"): {},
}

# Archetype strip: hand-picked chase artists and mid-range examples (each
# must still have 20+ mebounds), Reese's career, and the four lowest
# own-miss shares among 20+ mebound player-seasons (the paint anchors).
ARCH_PICKS = [
    ("WNBA", "2024", "Skylar Diggins"), ("NBA", "2024-25", "Jamal Murray"),
    ("NBA", "2023-24", "Brandon Miller"), ("NBA", "2025-26", "James Harden"),
    ("NBA", "2023-24", "Kyrie Irving"), ("NBA", "2025-26", "Luka Doncic"),
    ("NBA", "2024-25", "Jaylen Brown"), ("NBA", "2023-24", "Nikola Jokic"),
    ("NBA", "2024-25", "Alperen Sengun"),
]
ARCH_CHASE = [ARCH_PICKS[i] for i in (0, 1, 3, 4, 5)]  # the five the prose names: "59-75%"
ARCH_ANCHORS = 4
MIN_ME = 20          # archetype and chase/anchor lists
WNBA_MIN_REB = 200   # WNBA share ranking
NBA_MIN_REB = 300    # scatter
MIN_GAMES_RPG = 20   # per-game rebounding ranking
MIN_FG_MISSES = 100  # own-miss recovery ranking


# ---------------------------------------------------------------- formatting
def _q(x: Decimal, places: int) -> Decimal:
    return x.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def pct(num: int, den: int, places: int = 1) -> str:
    """Percent from exact counts, rounded half-up (never from a rounded share)."""
    return f"{_q(Decimal(num) * 100 / Decimal(den), places)}%"


def pct_range(pairs: list[tuple[int, int]]) -> str:
    vals = [_q(Decimal(n) * 100 / Decimal(d), 0) for n, d in pairs]
    return f"{min(vals)}–{max(vals)}%"


def comma(n: int) -> str:
    return f"{n:,}"


# ---------------------------------------------------------------- inputs
def load_player_seasons(path: Path = OUT / "mebounds_by_player_season.parquet") -> pl.DataFrame:
    return pl.read_parquet(path)


def ranked(df: pl.DataFrame) -> pl.DataFrame:
    """Count order, as in the top-mebounder CSVs: mebounds, then rebounds, then name."""
    return df.sort(["mebounds", "pbp_rebounds", "player"], descending=[True, True, False],
                   maintain_order=True)


def raw_counts() -> dict:
    """Counts no committed output holds, from data/ via the study's filters."""
    events = reb_rows = games = 0
    league_fg_miss: dict[str, int] = {}
    wnba_fg_miss = []
    for league, years in (("wnba", (2024, 2025, 2026)), ("nba", (2024, 2025, 2026))):
        for year in years:
            pbp = cm.load_pbp(league, year)
            label = f"{year - 1}-{str(year)[2:]}" if league == "nba" else str(year)
            events += pbp.height
            games += pbp["game_id"].n_unique()
            reb_rows += pbp.filter(pl.col("type_id").is_in([cm.OREB, cm.DREB])).height
            miss = pbp.filter(
                pl.col("shooting_play") & ~pl.col("scoring_play")
                & ~pl.col("type_text").str.contains("Free Throw")
                & pl.col("athlete_id_1").is_not_null())
            league_fg_miss[f"{league.upper()} {label}"] = miss.height
            if league == "wnba":
                wnba_fg_miss.append(miss.group_by("athlete_id_1").agg(pl.len().alias("fg_miss"))
                                    .with_columns(pl.lit(label).alias("season")))
    per = pl.concat(wnba_fg_miss).rename({"athlete_id_1": "athlete_id"})
    return {"events": events, "reb_rows": reb_rows, "games": games,
            "league_fg_miss": league_fg_miss, "wnba_fg_miss": per}


# ---------------------------------------------------------------- story_data
def _rows(df: pl.DataFrame, cols: list[str]) -> list[dict]:
    return df.select(cols).to_dicts()


def story_tables(ps: pl.DataFrame) -> dict:
    """The six analysis lists, in the July 28 file's schema and order."""
    lead_cols = ["season", "player", "team", "mebounds", "mebound_share_of_reb",
                 "mebound_share_of_oreb", "pbp_orebs"]
    reese = (ps.filter(pl.col("player") == REESE)
             .with_columns(dreb=pl.col("pbp_rebounds") - pl.col("pbp_orebs"),
                           other_oreb=pl.col("pbp_orebs") - pl.col("mebounds")))
    out = {"reese": _rows(reese, ["season", "team", "games", "pbp_rebounds", "pbp_orebs", "dreb",
                                  "other_oreb", "mebounds", "mebound_share_of_reb",
                                  "mebound_share_of_oreb"])}
    for league, key in (("WNBA", "wnba_top"), ("NBA", "nba_top")):
        top = pl.concat([ranked(ps.filter(pl.col("season") == s)).head(6) for s in SEASONS[league]])
        out[key] = _rows(top, lead_cols)
    sc = (pl.concat([ranked(ps.filter(pl.col("season") == s)) for s in SEASONS["NBA"]])
          .filter(pl.col("pbp_rebounds") >= NBA_MIN_REB)
          .with_columns(rpg=(pl.col("pbp_rebounds") / pl.col("games")).round(2)))
    out["nba_scatter"] = _rows(sc, ["season", "player", "team", "games", "pbp_rebounds", "rpg",
                                    "mebounds", "mebound_share_of_reb", "mebound_share_of_oreb"])
    q = ps.filter(pl.col("mebounds") >= MIN_ME)
    cc = ["league", "season", "player", "team", "mebounds", "pbp_orebs", "mebound_share_of_oreb"]
    out["chase"] = _rows(q.sort("mebound_share_of_oreb", descending=True, maintain_order=True)
                         .head(10), cc)
    out["anchor"] = _rows(q.sort("mebound_share_of_oreb", maintain_order=True).head(10), cc)
    return out


def ps_row(ps: pl.DataFrame, league: str, season: str, player: str) -> dict:
    r = ps.filter((pl.col("league") == league) & (pl.col("season") == season)
                  & (pl.col("player") == player))
    assert r.height == 1, (league, season, player, r.height)
    return r.row(0, named=True)


def context(ps: pl.DataFrame, raw: dict) -> dict:
    """Derived tables the facts and claims read."""
    w = ps.filter(pl.col("league") == "WNBA")
    n = ps.filter(pl.col("league") == "NBA")
    reese = {r["season"]: r for r in ps.filter(pl.col("player") == REESE).to_dicts()}
    top5 = {s: ranked(ps.filter(pl.col("season") == s)).head(5).to_dicts()
            for lg in SEASONS for s in SEASONS[lg]}
    rec = (w.join(raw["wnba_fg_miss"], on=["athlete_id", "season"], how="left")
           .with_columns(pl.col("fg_miss").fill_null(0)))
    cross = pl.read_csv(OUT / "cross_feed_comparison.csv")
    return {"ps": ps, "w": w, "n": n, "reese": reese, "top5": top5, "rec": rec,
            "raw": raw, "cross": cross}


def facts(c: dict) -> dict[str, str]:
    ps, w, n, R, raw = c["ps"], c["w"], c["n"], c["reese"], c["raw"]
    S = SEASONS["WNBA"]
    rme = sum(R[s]["mebounds"] for s in S)
    rreb = sum(R[s]["pbp_rebounds"] for s in S)
    rec = c["rec"]
    rfg = {s: rec.filter((pl.col("player") == REESE) & (pl.col("season") == s))["fg_miss"][0]
           for s in S}
    wnba_fg = sum(raw["league_fg_miss"][f"WNBA {s}"] for s in S)
    f: dict[str, str] = {
        "exhibition_games": str(sum(len(v) for v in cm.EXHIBITION_GAME_IDS.values())),
        "games_total": comma(int(c["cross"]["games_compared"].sum())),
        "rebounds_total": comma(raw["reb_rows"]),
        "career_mebounds": str(rme),
        "career_reb": comma(rreb),
        "career_share_reb": pct(rme, rreb),
        "career_share_reb_num": pct(rme, rreb).rstrip("%"),
        "one_in": str(_q(Decimal(rreb) / Decimal(rme), 0)),
        "wnba_share_reb": pct(w["mebounds"].sum(), w["pbp_rebounds"].sum()),
        "wnba_oreb": pct(w["mebounds"].sum(), w["pbp_orebs"].sum()),
        "nba_share_reb": pct(n["mebounds"].sum(), n["pbp_rebounds"].sum()),
        "nba_oreb": pct(n["mebounds"].sum(), n["pbp_orebs"].sum()),
        "rpg_2024": str(_q(Decimal(R["2024"]["pbp_rebounds"]) / R["2024"]["games"], 1)),
        "recov_range": pct_range([(R[s]["mebounds"], rfg[s]) for s in S]),
        "wnba_recov": pct(w["mebounds"].sum(), wnba_fg, 0),
        "recov_2026_int": pct(R["2026"]["mebounds"], rfg["2026"], 0),
        "other_oreb_2025": str(R["2025"]["pbp_orebs"] - R["2025"]["mebounds"]),
        "other_oreb_2026": str(R["2026"]["pbp_orebs"] - R["2026"]["mebounds"]),
        "reese_oreb_range": pct_range([(R[s]["mebounds"], R[s]["pbp_orebs"]) for s in S]),
        "wnba_200_count": str(w.filter(pl.col("pbp_rebounds") >= WNBA_MIN_REB).height),
        "nba_300_count": str(n.filter(pl.col("pbp_rebounds") >= NBA_MIN_REB).height),
        "nba_over10_count": str(n.filter((pl.col("pbp_rebounds") >= NBA_MIN_REB)
                                         & (pl.col("mebounds") / pl.col("pbp_rebounds") > 0.10)).height),
    }
    for s in S:
        r = R[s]
        f[f"games_{s}"] = str(r["games"])
        f[f"reb_{s}"] = str(r["pbp_rebounds"])
        f[f"me_{s}"] = str(r["mebounds"])
        f[f"share_reb_{s}"] = pct(r["mebounds"], r["pbp_rebounds"])
        f[f"oreb_share_{s}"] = pct(r["mebounds"], r["pbp_orebs"])
        f[f"recov_{s}"] = pct(r["mebounds"], rfg[s])
        f[f"runner_{s}"] = str(c["top5"][s][1]["mebounds"])
    # The page prints only some per-season values; an unprinted fact would
    # fail the "never printed" check, so drop the rest here.
    for k in ("games_2025", "games_2026", "reb_2025", "reb_2026", "me_2025", "runner_2025"):
        del f[k]
    ew = ps_row(ps, "WNBA", "2025", "Elizabeth Williams")
    f |= {"williams_me": str(ew["mebounds"]), "williams_reb": str(ew["pbp_rebounds"]),
          "williams_share": pct(ew["mebounds"], ew["pbp_rebounds"])}
    aw = ps_row(ps, "WNBA", "2024", "A'ja Wilson")
    f["wilson_oreb_2024"] = pct(aw["mebounds"], aw["pbp_orebs"])
    az = ps_row(ps, "WNBA", "2026", "Azura Stevens")
    f["stevens_oreb_2026"] = pct(az["mebounds"], az["pbp_orebs"], 0)
    f["chase_range"] = pct_range([(r["mebounds"], r["pbp_orebs"])
                                  for r in (ps_row(ps, *k) for k in ARCH_CHASE)])
    g = ps_row(ps, "NBA", "2024-25", "Rudy Gobert")
    f["gobert_share"] = pct(g["mebounds"], g["pbp_orebs"], 0)
    jk = n.filter(pl.col("player") == "Nikola Jokic")
    f["jokic_career_oreb"] = pct(jk["mebounds"].sum(), jk["pbp_orebs"].sum())
    f["jokic_me_total"] = str(jk["mebounds"].sum())
    for s in SEASONS["NBA"]:
        j = ps_row(ps, "NBA", s, "Nikola Jokic")
        f[f"jokic_me_{s}"] = str(j["mebounds"])
        f[f"jokic_oreb_{s}"] = pct(j["mebounds"], j["pbp_orebs"])
    f["sengun_two"] = str(sum(ps_row(ps, "NBA", s, "Alperen Sengun")["mebounds"]
                              for s in ("2024-25", "2025-26")))
    z = ps_row(ps, "NBA", "2023-24", "Zion Williamson")
    f["zion_share"] = pct(z["mebounds"], z["pbp_rebounds"], 2)
    for key, s, p in (("capela_oreb", "2023-24", "Clint Capela"),
                      ("zubac_oreb", "2024-25", "Ivica Zubac"),
                      ("clingan_oreb", "2025-26", "Donovan Clingan")):
        r = ps_row(ps, "NBA", s, p)
        f[key] = pct(r["mebounds"], r["pbp_orebs"])
    f["giannis_me_2024-25"] = str(ps_row(ps, "NBA", "2024-25", "Giannis Antetokounmpo")["mebounds"])
    f["towns_me_2025-26"] = str(ps_row(ps, "NBA", "2025-26", "Karl-Anthony Towns")["mebounds"])
    box = n.group_by("season").agg(pl.col("pbp_rebounds").sum().cast(pl.Int64), pl.col("box_rebounds").sum())
    worst = max(Decimal(abs(a - b)) * 100 / Decimal(b) for a, b in box.select("pbp_rebounds", "box_rebounds").rows())
    f["nba_box_diff"] = f"{_q(worst, 2)}%"
    return dict(sorted(f.items()))


def league_lines(c: dict) -> dict:
    n = c["n"]
    return {"nba_share_of_reb": round(n["mebounds"].sum() / n["pbp_rebounds"].sum(), 4),
            "nba_share_of_oreb": round(n["mebounds"].sum() / n["pbp_orebs"].sum(), 4)}


def page_data(tables: dict, c: dict) -> dict:
    """The chart data the page embeds as `const DATA`."""
    def nm(p: str) -> str:
        return DISPLAY_NAMES.get(p, p)

    def top5(rows: list[dict]) -> list[dict]:
        out, seen = [], {}
        for r in rows:
            seen[r["season"]] = seen.get(r["season"], 0) + 1
            if seen[r["season"]] <= 5:
                out.append({**r, "player": nm(r["player"])})
        return out

    scatter = []
    for r in tables["nba_scatter"]:
        d = {"player": nm(r["player"]), "team": r["team"], "season": r["season"], "rpg": r["rpg"],
             "mebounds": r["mebounds"], "pbp_rebounds": r["pbp_rebounds"],
             "mebound_share_of_reb": r["mebound_share_of_reb"],
             "mebound_share_of_oreb": r["mebound_share_of_oreb"]}
        hl = SCATTER_HL.get((r["player"], r["season"]))
        if hl is not None:
            d["hl"] = 1
            if hl.get("lab") == "zion":
                hl = {**hl, "lab": "Zion " + c["facts"]["zion_share"]}
            d |= hl
        scatter.append(d)
    assert sum(1 for d in scatter if d.get("hl")) == len(SCATTER_HL), "scatter highlight missing"

    ps = c["ps"]
    arch = []
    for k in ARCH_PICKS:
        r = ps_row(ps, *k)
        assert r["mebounds"] >= MIN_ME, (k, r["mebounds"])
        arch.append(r)
    anchors = (ps.filter(pl.col("mebounds") >= MIN_ME)
               .sort("mebound_share_of_oreb", maintain_order=True).head(ARCH_ANCHORS).to_dicts())
    R = c["reese"]
    rme = sum(r["mebounds"] for r in R.values())
    rore = sum(r["pbp_orebs"] for r in R.values())
    career = {"player": REESE, "league": "WNBA", "season": "2024-26 career",
              "mebounds": rme, "pbp_orebs": rore,
              "mebound_share_of_oreb": float(_q(Decimal(rme) / Decimal(rore), 4))}
    rows = [{"player": nm(r["player"]), "league": r["league"], "season": r["season"],
             "mebounds": r["mebounds"], "orebs": r["pbp_orebs"],
             "share": r["mebound_share_of_oreb"]} for r in arch + [career] + anchors]
    rows.sort(key=lambda d: -d["share"])  # stable: ties keep pick order
    return {"reese": tables["reese"], "wnba_top": top5(tables["wnba_top"]),
            "nba_top": top5(tables["nba_top"]), "nba_scatter": scatter, "arch": rows,
            "league": league_lines(c)}


def build() -> tuple[dict, dict, dict]:
    ps = load_player_seasons()
    raw = raw_counts()
    c = context(ps, raw)
    assert raw["games"] == int(c["cross"]["games_compared"].sum()), (
        "data/ game count disagrees with cross_feed_comparison.csv", raw["games"])
    f = facts(c)
    c["facts"] = f
    tables = story_tables(ps)
    story = tables | {"facts": f}
    return story, page_data(tables, c), c


def story_json(story: dict) -> str:
    return json.dumps(story, indent=1, ensure_ascii=False)


def data_line(data: dict) -> str:
    return "const DATA = " + json.dumps(data, ensure_ascii=False) + ";"


def with_data(page: str, line: str) -> str:
    lines = page.split("\n")
    hits = [i for i, x in enumerate(lines) if x.startswith("const DATA = ")]
    assert len(hits) == 1, f"expected one DATA line, found {len(hits)}"
    lines[hits[0]] = line
    return "\n".join(lines)


# ---------------------------------------------------------------- claims
def _rank1(df: pl.DataFrame, col: str) -> str:
    return df.sort(col, descending=True, maintain_order=True)["player"][0]


def claims(c: dict) -> list[tuple[str, bool]]:
    """(sentence fragment that must appear in the page's text, holds on outputs)."""
    w, n, R, top5, rec, raw = c["w"], c["n"], c["reese"], c["top5"], c["rec"], c["raw"]
    S, NS = SEASONS["WNBA"], SEASONS["NBA"]
    ws = {s: w.filter(pl.col("season") == s) for s in S}
    share = {s: ws[s].filter(pl.col("pbp_rebounds") >= WNBA_MIN_REB)
             .with_columns(x=pl.col("mebounds") / pl.col("pbp_rebounds")) for s in S}
    oreb = lambda r: r["mebounds"] / r["pbp_orebs"]  # noqa: E731
    reb = lambda r: r["mebounds"] / r["pbp_rebounds"]  # noqa: E731
    rr = lambda lg, s, p: ps_row(c["ps"], lg, s, p)  # noqa: E731
    rpg24 = ws["2024"].filter(pl.col("games") >= MIN_GAMES_RPG).with_columns(
        x=pl.col("pbp_rebounds") / pl.col("games"))
    rec26 = rec.filter((pl.col("season") == "2026") & (pl.col("fg_miss") >= MIN_FG_MISSES)).with_columns(
        x=pl.col("mebounds") / pl.col("fg_miss"))
    top6_25 = ranked(ws["2025"]).head(6)
    second26 = ws["2026"].filter(pl.col("mebounds") == top5["2026"][1]["mebounds"])["player"].to_list()
    lead_pg = {s: (R[s]["mebounds"] - top5[s][1]["mebounds"]) / R[s]["games"] for s in S}
    in_all_top5 = set.intersection(*(set(d["player"] for d in top5[s]) for s in S))
    sc = n.filter(pl.col("pbp_rebounds") >= NBA_MIN_REB).with_columns(
        x=pl.col("mebounds") / pl.col("pbp_rebounds"))
    above_reese = sc.filter(pl.col("x") > reb(R["2024"])).select("player", "season").rows()
    z = rr("NBA", "2023-24", "Zion Williamson")
    runners = [top5[s][1]["player"] for s in NS]
    jk = n.filter(pl.col("player") == "Nikola Jokic")
    gob = n.filter(pl.col("player") == "Rudy Gobert")["pbp_orebs"]
    arch = c.get("arch_rows", [])
    wbox = c["w"].group_by("season").agg(pl.col("pbp_rebounds").sum().cast(pl.Int64),
                                         pl.col("box_rebounds").sum())
    return [
        ("just over two million play-by-play events", 2_000_000 <= raw["events"] < 2_100_000),
        ("In mebound count, 2024, 2025 & 2026; in rate, 2024 & 2026",
         all(top5[s][0]["player"] == REESE for s in S)
         and [s for s in S if _rank1(share[s], "x") == REESE] == ["2024", "2026"]),
        ("A third of her offensive rebounds that year were her own misses",
         0.30 <= oreb(R["2024"]) <= 0.37),
        ("the most per game in the WNBA", _rank1(rpg24, "x") == REESE),
        ("while still converting a league-best", _rank1(rec26, "x") == REESE),
        ("She has never once rebounded her own missed free throw",
         sum(r["mebounds_ft"] for r in R.values()) == 0),
        ("the runner-up, Aliyah Boston", top5["2024"][1]["player"] == "Aliyah Boston"),
        ("That is a two-to-one gap",
         1.9 <= R["2024"]["mebounds"] / top5["2024"][1]["mebounds"] <= 2.1),
        ("In 2025 the runner-up was her own teammate",
         top5["2025"][1]["team"] == R["2025"]["team"]),
        ("Chicago put three players in the league's top six (Reese, Kamilla Cardoso, Elizabeth Williams)",
         sorted(top6_25.filter(pl.col("team") == "CHI")["player"].to_list())
         == sorted([REESE, "Kamilla Cardoso", "Elizabeth Williams"])),
        ("over a three-way tie of Breanna Stewart, Kamilla Cardoso, and Azura Stevens",
         sorted(second26) == ["Azura Stevens", "Breanna Stewart", "Kamilla Cardoso"]),
        ("She still leads the WNBA in mebound count every season, but no longer in mebound share every season",
         all(top5[s][0]["player"] == REESE for s in S)
         and not all(_rank1(share[s], "x") == REESE for s in S)),
        ("and both feeds agree on both players",
         pl.read_csv(OUT / "cross_feed_players.csv").filter(
             (pl.col("season") == "2025") & pl.col("player").is_in([REESE, "Elizabeth Williams"]))
         ["diff"].to_list() == [0, 0]),
        ("By count the trademark holds every season; by rate, two seasons of three",
         all(top5[s][0]["player"] == REESE for s in S)
         and sum(_rank1(share[s], "x") == REESE for s in S) == 2),
        ("she ranks second in total rebounds each season rather than leading the league",
         all(ws[s].sort("pbp_rebounds", descending=True)["player"].to_list().index(REESE) == 1
             for s in S)),
        ("her 2026 lead in mebound count was never her largest relative to games played",
         lead_pg["2026"] < max(lead_pg.values())),
        ("the highest mebound share of total rebounds is hers in 2024 and 2026",
         [s for s in S if _rank1(share[s], "x") == REESE] == ["2024", "2026"]),
        ("her Chicago teammate Elizabeth Williams edged her",
         _rank1(share["2025"], "x") == "Elizabeth Williams"
         and rr("WNBA", "2025", "Elizabeth Williams")["team"] == "CHI"),
        ("In 2025 Reese tops that view too",
         max(top5["2025"], key=oreb)["player"] == REESE),
        ("in 2024 only A'ja Wilson's",
         [d["player"] for d in top5["2024"] if oreb(d) > oreb(R["2024"])] == ["A'ja Wilson"]),
        ("on barely half of Reese's mebound volume",
         0.5 <= rr("WNBA", "2026", "Azura Stevens")["mebounds"] / R["2026"]["mebounds"] < 0.6),
        ("Reese's share sits at a career-low",
         min(S, key=lambda s: oreb(R[s])) == "2026"),
        ("the only WNBA player in the count top five all three seasons", in_all_top5 == {REESE}),
        ("most mebounds in the league three years running",
         all(top5[s][0]["player"] == "Nikola Jokic" for s in NS)),
        ("they're only ~9% of his enormous haul",
         round(100 * jk["mebounds"].sum() / jk["pbp_rebounds"].sum()) == 9),
        ("own the rate corner: 12–13% of their rebounds are self-service",
         all(0.12 <= reb(rr("NBA", s, p)) < 0.135 for s, p in (
             ("2023-24", "Zion Williamson"), ("2024-25", "Devin Booker"), ("2024-25", "Jaylen Brown")))),
        ("Over half of Brown's and Booker's offensive boards are their own misses",
         all(oreb(rr("NBA", "2024-25", p)) > 0.5 for p in ("Jaylen Brown", "Devin Booker"))),
        ("almost nobody escapes it upward past 10%",
         sc.filter(pl.col("x") > 0.10).height / sc.height < 0.06),
        ("Paint anchors like Rudy Gobert live at the bottom",
         bool(arch) and arch[-1]["player"] == "Rudy Gobert"),
        ("almost 300 offensive boards a season", 250 <= gob.mean() < 300),
        ("Reese sits a third of the way up",
         0.28 <= sum(r["mebounds"] for r in R.values()) / sum(r["pbp_orebs"] for r in R.values()) <= 0.38),
        ("Alperen Şengün is the only repeat runner-up",
         runners.count("Alperen Sengun") >= 2
         and all(runners.count(p) < 2 for p in set(runners) - {"Alperen Sengun"})),
        ("tops every NBA player-season in the sample except",
         above_reese == [("Zion Williamson", "2023-24")]),
        ("And Zion did it on half her per-game rebounding",
         0.4 <= (z["pbp_rebounds"] / z["games"]) / (R["2024"]["pbp_rebounds"] / R["2024"]["games"]) <= 0.6),
        ("Jokić also posts the best own-miss share",
         all(max(top5[s], key=oreb)["player"] == "Nikola Jokic" for s in NS)),
        ("Aaron Gordon, Giannis, and Scottie Barnes hold shares in the thirties",
         all(0.30 <= oreb(rr("NBA", s, p)) < 0.40 for s, p in (
             ("2023-24", "Aaron Gordon"), ("2024-25", "Giannis Antetokounmpo"),
             ("2025-26", "Scottie Barnes")))
         and all(any(d["player"] == p for d in top5[s]) for s, p in (
             ("2023-24", "Aaron Gordon"), ("2024-25", "Giannis Antetokounmpo"),
             ("2025-26", "Scottie Barnes")))),
        ("only about one rebound in twenty is a mebound",
         all(0.045 <= df["mebounds"].sum() / df["pbp_rebounds"].sum() < 0.065 for df in (w, n))),
        ("ranking among the league's top two rebounders every season",
         all(REESE in ws[s].sort("pbp_rebounds", descending=True).head(2)["player"].to_list()
             for s in S)),
        ("within 1 rebound league-wide per WNBA season",
         all(abs(a - b) <= 1 for a, b in wbox.select("pbp_rebounds", "box_rebounds").rows())),
    ]


# ---------------------------------------------------------------- page check
FACT_RE = re.compile(r'<(\w+)\b[^>]*\bdata-fact="([^"]+)"[^>]*>([^<]*)</\1>')
CORRECTION_RE = re.compile(r"<p\b[^>]*\bdata-correction\b.*?</p>", re.S)
NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

# Untagged numbers allowed in the prose. A bare token is allowed anywhere
# (season years, fixed thresholds); a context phrase allows only the numbers
# inside that exact phrase, and each names what checks it.
NUMBER_TOKENS = {
    "2023": "season 2023-24", "2024": "season", "2025": "season", "2026": "season",
    "24": "season suffix", "26": "season suffix",
    "300": "scatter threshold (NBA_MIN_REB)", "200": "WNBA share threshold (WNBA_MIN_REB)",
    "20": "archetype threshold (MIN_ME)",
}
NUMBER_CONTEXTS = {
    "July 25": "data cutoff date (README)",
    "WNBA rank, every season #1": "tile; claim 'In mebound count, 2024, 2025 & 2026; ...'",
    "upward past 10%": "the 10% line; claim 'almost nobody escapes it upward past 10%'",
    "12–13% of their rebounds": "claim 'own the rate corner: 12–13% ...'",
    "~9% of his enormous haul": "claim 'they're only ~9% of his enormous haul'",
    "≈ 1 of every": "the 1 in '1 of every 8' (the 8 is fact one_in)",
    "within 1 rebound": "claim 'within 1 rebound league-wide per WNBA season'",
}


def page_text(fragment: str) -> str:
    t = re.sub(r"<(script|style)\b.*?</\1>", " ", fragment, flags=re.S)
    t = re.sub(r"<[^>]+>", "", t)
    return re.sub(r"\s+", " ", html.unescape(t)).replace("​", "")


def check_page(page: str, f: dict, cl: list[tuple[str, bool]]) -> list[str]:
    errs = []
    used = set()
    for _, key, val in FACT_RE.findall(page):
        used.add(key)
        if key not in f:
            errs.append(f"data-fact {key!r} is not a known fact")
        elif html.unescape(val).strip() != f[key]:
            errs.append(f"data-fact {key!r}: page prints {html.unescape(val)!r}, outputs give {f[key]!r}")
    for key in sorted(set(f) - used):
        errs.append(f"fact {key!r} is never printed on the page (drop it or tag it)")
    text = page_text(page)
    for phrase, ok in cl:
        if phrase not in text:
            errs.append(f"claim sentence missing from page: {phrase!r}")
        elif not ok:
            errs.append(f"claim no longer holds on outputs: {phrase!r}")
    body = page[page.index("<body"):] if "<body" in page else page
    body = CORRECTION_RE.sub(" ", body)
    body = re.sub(r"<script\b.*?</script>", " ", body, flags=re.S)
    body = FACT_RE.sub(" ", body)
    body = re.sub(r"&#(\d+);", " ", body)
    btext = page_text(body)
    allowed = set()
    for phrase in NUMBER_CONTEXTS:
        for m in re.finditer(re.escape(phrase), btext):
            allowed.update(range(m.start(), m.end()))
    for m in NUM_RE.finditer(btext):
        tok = m.group(0).rstrip(",.")
        if tok in NUMBER_TOKENS or m.start() in allowed:
            continue
        ctx = btext[max(0, m.start() - 40):m.end() + 20]
        errs.append(f"untagged number {tok!r} in prose: ...{ctx}...")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--inject", choices=["json", "prose", "untagged", "claim"])
    a = ap.parse_args()

    story, data, c = build()
    c["arch_rows"] = data["arch"]
    want_json = story_json(story)
    page = STORY.read_text()
    if not a.check:
        JSON_OUT.write_text(want_json)
        page = with_data(page, data_line(data))
        STORY.write_text(page)
        DOCS.write_text(page)
        print(f"wrote {JSON_OUT.relative_to(ROOT)}, story/index.html, docs/index.html")

    have_json = JSON_OUT.read_text()
    docs = DOCS.read_text()
    cl = claims(c)
    if a.inject == "json":
        have_json = have_json.replace('"career_mebounds": "132"', '"career_mebounds": "136"', 1)
        assert have_json != want_json
    elif a.inject == "prose":
        page = page.replace('data-fact="career_mebounds">132<', 'data-fact="career_mebounds">136<', 1)
    elif a.inject == "untagged":
        page = page.replace("Nobody in her league mebounds like her",
                            "Nobody in her league mebounds like her, 457 boards", 1)
    elif a.inject == "claim":
        cl = [(p, ok if i else False) for i, (p, ok) in enumerate(cl)]

    errs = []
    if have_json != want_json:
        errs.append("output/story_data.json does not regenerate byte-identically")
    if data_line(data) not in page.split("\n"):
        errs.append("story/index.html DATA line differs from the regenerated chart data")
    if docs != STORY.read_text():
        errs.append("docs/index.html differs from story/index.html")
    errs += check_page(page, story["facts"], cl)
    for e in errs:
        print("FAIL:", e)
    if errs:
        return 1
    print(f"ok: story_data.json regenerates; page DATA current; docs == story; "
          f"{len(story['facts'])} facts and {len(cl)} claims match outputs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
