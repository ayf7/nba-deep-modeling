# Data Workspace

Tracked code for data conversion and derived-table builds lives in `data/scripts`.

Converted and derived data lives under `data/artifacts` and is intentionally ignored by git.

The project uses three SQLite layers:

```text
nba_raw.sqlite       # one-to-one imports from external raw CSV archives
oddsportal_moneyline.sqlite
nba_core.sqlite      # cleaned, typed basketball tables
player_matchup_training.sqlite
features.sqlite      # model-facing labels/features
features_embeddings_v1.sqlite
```

## Layout

```text
data/
  scripts/
    convert_nba_archives.py
    build_nba_core.py
    build_features.py
  artifacts/        # ignored data artifacts
    nba_raw.sqlite
    oddsportal_moneyline.sqlite
    nba_core.sqlite
    player_matchup_training.sqlite
    features.sqlite
    features_embeddings_v1.sqlite
```

## NBA Data Conversion

The `shufinskiy/nba_data` submodule stores each dataset as a `.tar.xz` archive containing one CSV file. To convert selected NBA regular-season archives into a raw SQLite database:

```bash
python data/scripts/convert_nba_archives.py --seasons 2020 2021 2022 2023 2024 2025
```

This creates:

```text
data/artifacts/nba_raw.sqlite
```

By default it converts:

```text
cdnnba
nbastatsv3
shotdetail
matchups
```

Use `--sources` to restrict conversion:

```bash
python data/scripts/convert_nba_archives.py --seasons 2024 2025 --sources cdnnba shotdetail
```

## Core Tables

Build cleaned, typed basketball tables from the raw import:

```bash
python data/scripts/build_nba_core.py --seasons 2020 2021 2022 2023 2024 2025
```

This creates:

```text
data/artifacts/nba_core.sqlite
```

If `data/artifacts/oddsportal_moneyline.sqlite` exists, the core build also
imports matched moneyline odds into `game_moneyline_odds`. Use `--skip-odds`
to build basketball-only core tables.

Current core tables:

```text
games
teams
players
game_events
shots
player_matchups
game_moneyline_odds
```

## Player Matchup Training Rows

Build grouped player-defender interaction rows for embedding models:

```bash
python data/scripts/build_player_matchup_training.py
```

This creates:

```text
data/artifacts/player_matchup_training.sqlite
```

The output table `matchup_training_rows` groups `player_matchups` by game,
offensive player, defender, and offensive team. It uses `partial_possessions`
as `exposure_possessions` and sums matchup outcome counts including
`player_points`, field-goal attempts/makes, three-point attempts/makes,
turnovers, assists, potential assists, free throws, and shooting fouls.

Default filters keep rows with at least `0.5` partial possessions and `5`
matchup seconds. Use `--min-exposure-possessions` and `--min-matchup-seconds`
to change those thresholds.

## Feature Tables

Build the model-facing database:

```bash
python data/scripts/build_features.py
```

This creates:

```text
data/artifacts/features.sqlite
```

Current feature tables:

```text
feature_builds
game_labels
team_game_results
team_pregame_features
model_games
```

`game_labels` contains one row per game with the home-win label.

`team_game_results` contains two rows per game, one from each team's perspective.

`team_pregame_features` contains one row per game/team with features computed only from that team's previous games.

`model_games` is the wide training table with one row per game. It currently includes:

```text
game_id
season
game_date
home_team_id
away_team_id
label_home_win
home_games_played_before
away_games_played_before
home/away/diff win percentage before game
home/away/diff average point differential before game
home/away/diff average points for before game
home/away/diff average points against before game
home/away/diff rolling last-5 versions of those stats
home/away/diff rolling last-10 versions of those stats
home_rest_days
away_rest_days
diff_rest_days
```

Early-season rows intentionally have `NULL` rolling values when a team has no prior games. Modeling code should either filter warmup games or impute those values explicitly.

## Embedding-Augmented Feature Tables

After training player matchup embedding snapshots, build an augmented game
feature database:

```bash
python models_embedding/scripts/build_embedding_game_features.py
```

This creates:

```text
data/artifacts/features_embeddings_v1.sqlite
```

The output `model_games` table starts from `features.sqlite` and adds
leakage-safe player embedding matchup features. Each game receives the latest
available embedding snapshot whose training data ends before the game date.
Recent rotations are estimated from prior games only.

## OddsPortal Probe

The OddsPortal probe fetches NBA results-table moneyline odds from the public
archive endpoint and writes ignored artifacts under:

```text
data/artifacts/oddsportal_probe/
```

It also builds a consolidated SQLite database:

```text
data/artifacts/oddsportal_moneyline.sqlite
```

Fetch one page:

```bash
python data/scripts/probe_oddsportal.py --page 10
```

Fetch or resume all pages for the default season URL:

```bash
python data/scripts/probe_oddsportal.py --all-pages --sleep-seconds 1.0
```

Fetch or resume several NBA seasons:

```bash
python data/scripts/probe_oddsportal.py \
  --season-start-years 2020 2021 2022 2023 2024 2025 \
  --current-season-start-year 2025 \
  --all-pages \
  --sleep-seconds 1.0
```

The script is resumable. It writes each page JSON atomically as soon as it is
fetched, skips valid existing page files by default, retries failed requests
with exponential backoff, and rebuilds the flat moneyline CSV from saved page
JSON. When all pages are already cached, a normal `--all-pages` run skips
network page fetches and goes straight to rebuilding CSV/SQLite artifacts.

Useful controls:

```bash
# Smoke test the first two pages.
python data/scripts/probe_oddsportal.py --all-pages --max-pages 2

# Rebuild the CSV without making network requests.
python data/scripts/probe_oddsportal.py --rebuild-csv-only

# Rebuild only the consolidated SQLite database from existing CSVs.
python data/scripts/probe_oddsportal.py --rebuild-sqlite-only

# Refetch even when page JSON already exists.
python data/scripts/probe_oddsportal.py --all-pages --force
```

The SQLite table stores decimal odds as the canonical values:

```text
home_avg_decimal_odds
away_avg_decimal_odds
```

American odds and normalized market-implied probabilities are derived columns
for readability and model comparison.
