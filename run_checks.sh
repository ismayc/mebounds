#!/usr/bin/env bash
# Checks for the mebounds study. Run from anywhere; exits nonzero on failure.
#   1. recompute the ESPN-based study outputs
#   2. rebuild the cross-feed tables against the NBA's PlayByPlayV3 cache
#      (../playbyplay-study/data/raw_pbp) and gate them
#   3. prove the gate fails on the two known failure modes
#   4. check the story page against the outputs: story_data.json regenerates
#      byte-identically, the page's chart data is current, docs/ mirrors
#      story/, and every tagged number and listed claim matches the outputs
#      (python/build_story_data.py --check; rebuild with no flag)
#   5. prove the page check fails on each of its four failure modes
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python

$PY python/compute_mebounds.py > /dev/null
$PY python/cross_feed_check.py

for fault in order exhibitions; do
  if $PY python/cross_feed_check.py --inject "$fault" > /dev/null; then
    echo "FAIL: gate passed with injected fault '$fault'"
    exit 1
  fi
  echo "ok: gate fails with injected fault '$fault'"
done

$PY python/build_story_data.py --check

for fault in json prose untagged claim; do
  if $PY python/build_story_data.py --check --inject "$fault" > /dev/null; then
    echo "FAIL: page check passed with injected fault '$fault'"
    exit 1
  fi
  echo "ok: page check fails with injected fault '$fault'"
done
echo "ALL MEBOUNDS CHECKS PASSED"
