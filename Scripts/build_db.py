"""
Mets Stats Database loader

Pulls current Mets roster + player season stats directly from MLB's API 
The aim of this is to create a statistical database that can be used in a RAG implementation.

The data gathered is stored locally with no auto-update feature. To update the DB this
will need to be manually rerun with --update.

Usage:
    python build_db.py                      # Loads the cache if present, else builds it
    python build_db.py --update             # rebuilds the cache
    python build_db.py --season 2026        # target a specific season's data
"""

# =======================================================================================
# STEP 0: SET IMPORTS AND GLOBAL VARIABLES
# =======================================================================================
import argparse
import json
import os
from datetime import datetime
import statsapi
import pandas as pd

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "Database")
BATTING_FILE = os.path.join(DB_DIR, "mets_batting.json")
PITCHING_FILE = os.path.join(DB_DIR, "mets_pitching.json")
META_FILE = os.path.join(DB_DIR, "meta.json")

METS_TEAM_ID = 121  # MLB StatsAPI's fixed team ID for the Mets (stable, won't change)


def ensure_dir():
    os.makedirs(DB_DIR, exist_ok=True)


# =======================================================================================
# STEP 1: GET ROSTER, PULL PER-PLAYER STATS, SPLIT INTO BATTING/PITCHING, DUMP TO JSON
# =======================================================================================
def fetch_and_save(season: int):
    # Pull the active roster directly (structured JSON, not the formatted string
    # that statsapi.roster() returns)
    print(f"Fetching {season} Mets active roster...")
    roster_raw = statsapi.get(
        "team_roster",
        {"teamId": METS_TEAM_ID, "rosterType": "active", "season": season},
    )

    batting_rows = []
    pitching_rows = []

    for entry in roster_raw.get("roster", []):
        person_id = entry["person"]["id"]
        name = entry["person"]["fullName"]
        position = entry["position"]["abbreviation"]

        print(f"  Pulling season stats for {name} ({position})...")
        data = statsapi.player_stat_data(
            person_id, group="[hitting,pitching]", type="season", season=season
        )

        for stat_group in data.get("stats", []):
            stats = stat_group["stats"]
            if stat_group["group"] == "hitting" and stats.get("atBats", 0):
                batting_rows.append({
                    "Name": name,
                    "Position": position,
                    "AVG": stats.get("avg"),
                    "OBP": stats.get("obp"),
                    "SLG": stats.get("slg"),
                    "HR": stats.get("homeRuns"),
                    "RBI": stats.get("rbi"),
                    "SB": stats.get("stolenBases"),
                    "AtBats": stats.get("atBats"),
                })
            elif stat_group["group"] == "pitching" and stats.get("inningsPitched"):
                pitching_rows.append({
                    "Name": name,
                    "Position": position,
                    "ERA": stats.get("era"),
                    "WHIP": stats.get("whip"),
                    "W": stats.get("wins"),
                    "L": stats.get("losses"),
                    "SV": stats.get("saves"),
                    "SO": stats.get("strikeOuts"),
                    "InningsPitched": stats.get("inningsPitched"),
                })

    ensure_dir()
    with open(BATTING_FILE, "w") as f:
        json.dump(batting_rows, f, indent=2)
    with open(PITCHING_FILE, "w") as f:
        json.dump(pitching_rows, f, indent=2)

    meta = {
        "season": season,
        "last_updated": datetime.now().isoformat(),
        "batting_rows": len(batting_rows),
        "pitching_rows": len(pitching_rows),
        "source": "MLB StatsAPI (statsapi.mlb.com)",
    }
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {len(batting_rows)} batting rows and {len(pitching_rows)} pitching rows.")
    return batting_rows, pitching_rows


# =======================================================================================
# STEP 2: LOAD DATABASE FOR RETRIEVAL, HANDLE ARGPARSE ARGUMENTS
# =======================================================================================
def load_database(season: int = None, force_update: bool = False):
    season = season or datetime.now().year

    cache_exists = os.path.exists(BATTING_FILE) and os.path.exists(PITCHING_FILE)

    if force_update or not cache_exists:
        reason = "forced update" if force_update else "no cache found"
        print(f"Building database ({reason})...")
        fetch_and_save(season)
    else:
        print(f"Loading cached database from {DB_DIR} (no network call)...")

    batting_df = pd.read_json(BATTING_FILE)
    pitching_df = pd.read_json(PITCHING_FILE)
    return batting_df, pitching_df


# =======================================================================================
# STEP 3: MAIN
# =======================================================================================
def main():
    parser = argparse.ArgumentParser(description="Build/load the Mets stats JSON database.")
    parser.add_argument("--update", action="store_true", help="Force a full rebuild")
    parser.add_argument("--season", type=int, default=None, help="Season year (default: current year).")
    args = parser.parse_args()

    batting_df, pitching_df = load_database(season=args.season, force_update=args.update)

    print("\nSample batting rows:")
    print(batting_df[["Name", "HR", "AVG", "OBP"]].head() if len(batting_df) else "No rows returned.")
    print("\nSample pitching rows:")
    print(pitching_df[["Name", "ERA", "WHIP"]].head() if len(pitching_df) else "No rows returned.")


if __name__ == "__main__":
    main()