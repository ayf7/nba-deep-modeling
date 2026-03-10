#!/usr/bin/env python3
"""Probe OddsPortal NBA archive moneyline odds.

This is intentionally a probe, not a production ingestion job. It mirrors the
public page's archive request, decrypts the response, and writes page-level raw
JSON plus a flat moneyline CSV for inspection.
"""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import html
import json
import random
import re
import sqlite3
import subprocess
import time
import urllib.parse
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


DEFAULT_URL = "https://www.oddsportal.com/basketball/usa/nba-2024-2025/results/"
CURRENT_NBA_URL = "https://www.oddsportal.com/basketball/usa/nba/results/"
DEFAULT_OUTPUT_DIR = Path("data/artifacts/oddsportal_probe")
DEFAULT_SQLITE_PATH = Path("data/artifacts/oddsportal_moneyline.sqlite")
TIMEZONE_PART_FALLBACKS = (-8, 8, -5, 0)
NY_TZ = ZoneInfo("America/New_York")
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "Chrome/124 Safari/537.36"
)

AES_PASSPHRASE = b"J*8sQ!p$7aD_fR2yW@gHn*3bVp#sAdLd_k"
AES_SALT = b"5b9a8f2c3e6d1a4b7c8e9d0f1a2b3c4d"


def make_opener() -> urllib.request.OpenerDirector:
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
    opener.addheaders = [
        ("User-Agent", USER_AGENT),
        ("Accept", "application/json, text/plain, */*"),
    ]
    return opener


MONEYLINE_FIELDS = [
    "event_id",
    "encoded_event_id",
    "date_start_timestamp",
    "home_name",
    "away_name",
    "home_result",
    "away_result",
    "result",
    "status_id",
    "event_stage_name",
    "tournament_name",
    "event_url",
    "home_avg_odds",
    "home_max_odds",
    "home_max_provider_id",
    "away_avg_odds",
    "away_max_odds",
    "away_max_provider_id",
]

SQLITE_COLUMNS = [
    "season_slug",
    "season_start_year",
    "season_end_year",
    "event_id",
    "encoded_event_id",
    "date_start_timestamp",
    "game_datetime_utc",
    "game_datetime_et",
    "game_date_et",
    "home_name",
    "away_name",
    "home_result",
    "away_result",
    "result",
    "status_id",
    "event_stage_name",
    "tournament_name",
    "event_url",
    "home_avg_decimal_odds",
    "away_avg_decimal_odds",
    "home_max_decimal_odds",
    "away_max_decimal_odds",
    "home_avg_american_odds",
    "away_avg_american_odds",
    "home_max_american_odds",
    "away_max_american_odds",
    "home_implied_prob_raw",
    "away_implied_prob_raw",
    "market_overround",
    "home_implied_prob_normalized",
    "away_implied_prob_normalized",
    "home_max_provider_id",
    "away_max_provider_id",
    "has_moneyline_odds",
]


def fetch_text(
    opener: urllib.request.OpenerDirector,
    url: str,
    *,
    referer: str | None = None,
    timeout_seconds: float = 60,
) -> str:
    headers = {}
    if referer:
        headers["Referer"] = referer
        headers["X-Requested-With"] = "XMLHttpRequest"
    request = urllib.request.Request(url, headers=headers)
    with opener.open(request, timeout=timeout_seconds) as response:
        return response.read().decode("utf-8")


def fetch_text_with_retries(
    opener: urllib.request.OpenerDirector,
    url: str,
    *,
    referer: str | None = None,
    timeout_seconds: float = 60,
    max_retries: int = 4,
    retry_base_seconds: float = 2.0,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return fetch_text(
                opener,
                url,
                referer=referer,
                timeout_seconds=timeout_seconds,
            )
        except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError) as error:
            last_error = error
            if attempt == max_retries:
                break
            sleep_seconds = retry_base_seconds * (2 ** (attempt - 1))
            sleep_seconds += random.uniform(0, retry_base_seconds)
            print(
                f"request failed on attempt {attempt}/{max_retries}; "
                f"sleeping {sleep_seconds:.1f}s before retry: {error}"
            )
            time.sleep(sleep_seconds)
    raise RuntimeError(f"failed to fetch {url}") from last_error


def extract_odds_request(page_html: str) -> dict[str, Any]:
    match = re.search(r':odds-request="([^"]+)"', page_html)
    if not match:
        raise ValueError("Could not find :odds-request in OddsPortal page HTML.")
    return json.loads(html.unescape(match.group(1)))


def extract_user_data_url(page_html: str, page_url: str) -> str:
    match = re.search(r'["\']([^"\']*/ajax-user-data/[^"\']+)["\']', page_html)
    if not match:
        raise ValueError("Could not find ajax-user-data script URL.")
    return urllib.parse.urljoin(page_url, html.unescape(match.group(1)))


def extract_page_var(user_data_js: str) -> dict[str, Any]:
    matches = re.findall(r'JSON\.parse\("((?:\\.|[^"\\])*)"\)', user_data_js)
    if not matches:
        raise ValueError("Could not find JSON.parse payload in ajax-user-data response.")
    # The first payload contains pageVar/userData; repeated payloads are fallbacks.
    return json.loads(json.loads(f'"{matches[0]}"'))


def build_archive_url(
    page_url: str,
    odds_request: dict[str, Any],
    page_var: dict[str, Any],
    page: int,
) -> str:
    root = urllib.parse.urljoin(page_url, odds_request["url"])
    bookiehash = page_var["bookiehash"]
    use_premium = page_var["usePremium"]
    timezone_part = odds_request["urlPartTz"]
    if page <= 1:
        return (
            f"{root}{bookiehash}/{use_premium}/{timezone_part}/"
            f"{odds_request['urlPartQs']}{int(time.time() * 1000)}"
        )
    return (
        f"{root}{bookiehash}/{use_premium}/{timezone_part}"
        f"?page={page}&_={int(time.time() * 1000)}"
    )


def decrypt_payload(encrypted_text: str) -> dict[str, Any]:
    outer = base64.b64decode(encrypted_text.strip()).decode("utf-8")
    ciphertext_b64, iv_hex = outer.split(":", 1)
    ciphertext = base64.b64decode(ciphertext_b64)
    key = hashlib.pbkdf2_hmac("sha256", AES_PASSPHRASE, AES_SALT, 1000, dklen=32)
    process = subprocess.run(
        [
            "openssl",
            "enc",
            "-aes-256-cbc",
            "-d",
            "-K",
            key.hex(),
            "-iv",
            iv_hex,
        ],
        input=ciphertext,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    plaintext = process.stdout
    if plaintext.startswith(b"\x1f\x8b"):
        plaintext = gzip.decompress(plaintext)
    return json.loads(plaintext)


def row_to_moneyline(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "event_id": row.get("id"),
        "encoded_event_id": row.get("encodeEventId"),
        "date_start_timestamp": row.get("date-start-timestamp"),
        "home_name": row.get("home-name"),
        "away_name": row.get("away-name"),
        "home_result": row.get("homeResult"),
        "away_result": row.get("awayResult"),
        "result": row.get("result"),
        "status_id": row.get("status-id"),
        "event_stage_name": row.get("event-stage-name"),
        "tournament_name": row.get("tournament-name"),
        "event_url": row.get("url"),
        "home_avg_odds": None,
        "home_max_odds": None,
        "home_max_provider_id": None,
        "away_avg_odds": None,
        "away_max_odds": None,
        "away_max_provider_id": None,
    }
    # OddsPortal's historical `outcomeResultId` is not a stable home/away
    # side identifier. On completed games it can reflect the result, so using
    # it as a side flips the odds for away wins. The odds list is emitted in
    # the same home/away order as the event row.
    for prefix, odds in zip(("home", "away"), row.get("odds") or []):
        out[f"{prefix}_avg_odds"] = odds.get("avgOdds")
        out[f"{prefix}_max_odds"] = odds.get("maxOdds")
        out[f"{prefix}_max_provider_id"] = odds.get("maxOddsProviderId")
    return out


def page_path(output_dir: Path, season_slug: str, page: int) -> Path:
    return output_dir / f"{season_slug}_page_{page}.json"


def csv_path(output_dir: Path, season_slug: str) -> Path:
    return output_dir / f"{season_slug}_moneyline.csv"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(path)


def load_page_file(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    data = payload.get("d")
    if not isinstance(data, dict):
        return None
    if not isinstance(data.get("rows"), list):
        return None
    return payload


def load_valid_page(output_dir: Path, season_slug: str, page: int) -> dict[str, Any] | None:
    path = page_path(output_dir, season_slug, page)
    payload = load_page_file(path)
    if payload is None:
        return None
    actual_page = payload.get("d", {}).get("page")
    if actual_page != page:
        return None
    return payload


def discover_saved_pages(output_dir: Path, season_slug: str) -> list[dict[str, Any]]:
    payloads = []
    for path in sorted(output_dir.glob(f"{season_slug}_page_*.json")):
        payload = load_page_file(path)
        if payload is not None:
            payloads.append(payload)
    return sorted(payloads, key=lambda payload: payload["d"].get("page", 0))


def filter_pages(
    pages: list[dict[str, Any]],
    allowed_pages: set[int] | None,
) -> list[dict[str, Any]]:
    if allowed_pages is None:
        return pages
    return [payload for payload in pages if payload["d"].get("page") in allowed_pages]


def write_moneyline_csv(
    output_dir: Path,
    season_slug: str,
    pages: list[dict[str, Any]],
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    moneyline_rows = []
    for payload in pages:
        for row in payload["d"].get("rows") or []:
            moneyline_rows.append(row_to_moneyline(row))

    with csv_path(output_dir, season_slug).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MONEYLINE_FIELDS)
        writer.writeheader()
        writer.writerows(moneyline_rows)
    return len(moneyline_rows)


def parse_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def decimal_to_american(decimal_odds: float | None) -> int | None:
    if decimal_odds is None or decimal_odds <= 1:
        return None
    if decimal_odds >= 2:
        return round((decimal_odds - 1) * 100)
    return round(-100 / (decimal_odds - 1))


def season_years_from_slug(season_slug: str) -> tuple[int | None, int | None]:
    match = re.search(r"(\d{4})-(\d{4})", season_slug)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def timestamp_fields(timestamp: int | None) -> tuple[str | None, str | None, str | None]:
    if timestamp is None:
        return None, None, None
    utc_dt = datetime.fromtimestamp(timestamp, timezone.utc)
    et_dt = utc_dt.astimezone(NY_TZ)
    return utc_dt.isoformat(), et_dt.isoformat(), et_dt.date().isoformat()


def sqlite_row_from_csv_row(season_slug: str, row: dict[str, str]) -> dict[str, Any]:
    season_start, season_end = season_years_from_slug(season_slug)
    timestamp = parse_int(row.get("date_start_timestamp"))
    game_datetime_utc, game_datetime_et, game_date_et = timestamp_fields(timestamp)

    home_avg = parse_float(row.get("home_avg_odds"))
    away_avg = parse_float(row.get("away_avg_odds"))
    home_max = parse_float(row.get("home_max_odds"))
    away_max = parse_float(row.get("away_max_odds"))
    home_raw = 1 / home_avg if home_avg else None
    away_raw = 1 / away_avg if away_avg else None
    raw_sum = None
    market_overround = None
    home_norm = None
    away_norm = None
    if home_raw is not None and away_raw is not None:
        raw_sum = home_raw + away_raw
        market_overround = raw_sum - 1
        home_norm = home_raw / raw_sum
        away_norm = away_raw / raw_sum

    has_moneyline = int(home_avg is not None and away_avg is not None)
    return {
        "season_slug": season_slug,
        "season_start_year": season_start,
        "season_end_year": season_end,
        "event_id": row.get("event_id") or None,
        "encoded_event_id": row.get("encoded_event_id") or None,
        "date_start_timestamp": timestamp,
        "game_datetime_utc": game_datetime_utc,
        "game_datetime_et": game_datetime_et,
        "game_date_et": game_date_et,
        "home_name": row.get("home_name") or None,
        "away_name": row.get("away_name") or None,
        "home_result": parse_int(row.get("home_result")),
        "away_result": parse_int(row.get("away_result")),
        "result": row.get("result") or None,
        "status_id": parse_int(row.get("status_id")),
        "event_stage_name": row.get("event_stage_name") or None,
        "tournament_name": row.get("tournament_name") or None,
        "event_url": row.get("event_url") or None,
        "home_avg_decimal_odds": home_avg,
        "away_avg_decimal_odds": away_avg,
        "home_max_decimal_odds": home_max,
        "away_max_decimal_odds": away_max,
        "home_avg_american_odds": decimal_to_american(home_avg),
        "away_avg_american_odds": decimal_to_american(away_avg),
        "home_max_american_odds": decimal_to_american(home_max),
        "away_max_american_odds": decimal_to_american(away_max),
        "home_implied_prob_raw": home_raw,
        "away_implied_prob_raw": away_raw,
        "market_overround": market_overround,
        "home_implied_prob_normalized": home_norm,
        "away_implied_prob_normalized": away_norm,
        "home_max_provider_id": parse_int(row.get("home_max_provider_id")),
        "away_max_provider_id": parse_int(row.get("away_max_provider_id")),
        "has_moneyline_odds": has_moneyline,
    }


def season_slug_from_csv_path(path: Path) -> str:
    return path.name.removesuffix("_moneyline.csv")


def build_moneyline_sqlite(output_dir: Path, sqlite_path: Path) -> dict[str, Any]:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    csv_paths = sorted(output_dir.glob("nba-*_moneyline.csv"))
    rows = []
    for path in csv_paths:
        season_slug = season_slug_from_csv_path(path)
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(sqlite_row_from_csv_row(season_slug, row))

    placeholders = ", ".join(f":{column}" for column in SQLITE_COLUMNS)
    with sqlite3.connect(sqlite_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(
            """
            DROP TABLE IF EXISTS moneyline_odds;

            CREATE TABLE moneyline_odds (
                season_slug TEXT NOT NULL,
                season_start_year INTEGER,
                season_end_year INTEGER,
                event_id TEXT NOT NULL,
                encoded_event_id TEXT,
                date_start_timestamp INTEGER,
                game_datetime_utc TEXT,
                game_datetime_et TEXT,
                game_date_et TEXT,
                home_name TEXT,
                away_name TEXT,
                home_result INTEGER,
                away_result INTEGER,
                result TEXT,
                status_id INTEGER,
                event_stage_name TEXT,
                tournament_name TEXT,
                event_url TEXT,
                home_avg_decimal_odds REAL,
                away_avg_decimal_odds REAL,
                home_max_decimal_odds REAL,
                away_max_decimal_odds REAL,
                home_avg_american_odds INTEGER,
                away_avg_american_odds INTEGER,
                home_max_american_odds INTEGER,
                away_max_american_odds INTEGER,
                home_implied_prob_raw REAL,
                away_implied_prob_raw REAL,
                market_overround REAL,
                home_implied_prob_normalized REAL,
                away_implied_prob_normalized REAL,
                home_max_provider_id INTEGER,
                away_max_provider_id INTEGER,
                has_moneyline_odds INTEGER NOT NULL,
                PRIMARY KEY (season_slug, event_id)
            );

            CREATE INDEX idx_moneyline_season_date
                ON moneyline_odds (season_start_year, game_date_et);
            CREATE INDEX idx_moneyline_home_away
                ON moneyline_odds (home_name, away_name);
            CREATE INDEX idx_moneyline_has_odds
                ON moneyline_odds (has_moneyline_odds);
            """
        )
        if rows:
            conn.executemany(
                f"""
                INSERT INTO moneyline_odds ({", ".join(SQLITE_COLUMNS)})
                VALUES ({placeholders})
                """,
                rows,
            )
        conn.commit()

    return {
        "sqlite_path": str(sqlite_path),
        "csv_files_loaded": len(csv_paths),
        "rows_loaded": len(rows),
        "rows_with_moneyline_odds": sum(row["has_moneyline_odds"] for row in rows),
    }


def fetch_archive_page(
    opener: urllib.request.OpenerDirector,
    page_url: str,
    odds_request: dict[str, Any],
    page_var: dict[str, Any],
    page: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    url = build_archive_url(page_url, odds_request, page_var, page)
    encrypted_text = fetch_text_with_retries(
        opener,
        url,
        referer=page_url,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        retry_base_seconds=args.retry_base_seconds,
    )
    payload = decrypt_payload(encrypted_text)
    actual_page = payload.get("d", {}).get("page")
    if actual_page != page:
        raise ValueError(f"requested page {page}, but payload reported page {actual_page}")
    return payload


def fetch_archive_page_with_timezone_fallback(
    opener: urllib.request.OpenerDirector,
    page_url: str,
    odds_request: dict[str, Any],
    page_var: dict[str, Any],
    page: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    attempted_timezone_parts = []
    candidate_timezone_parts = [odds_request.get("urlPartTz")]
    candidate_timezone_parts.extend(
        timezone_part
        for timezone_part in TIMEZONE_PART_FALLBACKS
        if timezone_part not in candidate_timezone_parts
    )
    last_error: Exception | None = None
    for timezone_part in candidate_timezone_parts:
        attempted_timezone_parts.append(timezone_part)
        candidate_odds_request = dict(odds_request)
        candidate_odds_request["urlPartTz"] = timezone_part
        try:
            payload = fetch_archive_page(
                opener,
                page_url,
                candidate_odds_request,
                page_var,
                page,
                args,
            )
            if timezone_part != odds_request.get("urlPartTz"):
                print(
                    f"page {page}: recovered with timezone part {timezone_part} "
                    f"after attempts {attempted_timezone_parts}"
                )
            return payload, candidate_odds_request
        except Exception as error:
            last_error = error
    raise RuntimeError(
        f"failed page {page} with timezone parts {attempted_timezone_parts}"
    ) from last_error


def season_slug_from_url(
    url: str,
    *,
    current_season_start_year: int | None = None,
) -> str:
    if current_season_start_year is not None and "/basketball/usa/nba/results/" in url:
        return f"nba-{current_season_start_year}-{current_season_start_year + 1}"
    parts = [part for part in urllib.parse.urlparse(url).path.split("/") if part]
    for part in parts:
        if part.startswith("nba"):
            return part
    return "oddsportal_nba"


def nba_season_url(start_year: int, *, current_start_year: int | None = None) -> str:
    if current_start_year is not None and start_year == current_start_year:
        return CURRENT_NBA_URL
    return f"https://www.oddsportal.com/basketball/usa/nba-{start_year}-{start_year + 1}/results/"


def target_urls_from_args(args: argparse.Namespace) -> list[str]:
    urls = list(args.urls or [])
    if args.season_start_years:
        urls.extend(
            nba_season_url(year, current_start_year=args.current_season_start_year)
            for year in args.season_start_years
        )
    if not urls:
        urls = [args.url]
    return list(dict.fromkeys(urls))


def cached_all_pages_available(output_dir: Path, season_slug: str) -> tuple[bool, int, list[dict[str, Any]]]:
    saved_pages = discover_saved_pages(output_dir, season_slug)
    if not saved_pages:
        return False, 0, []
    first_payload = load_valid_page(output_dir, season_slug, 1)
    if first_payload is None:
        return False, 0, saved_pages
    page_count = first_payload["d"].get("pagination", {}).get("pageCount", 1)
    available_pages = {payload["d"].get("page") for payload in saved_pages}
    expected_pages = set(range(1, page_count + 1))
    return expected_pages.issubset(available_pages), page_count, saved_pages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe OddsPortal NBA moneyline archive odds.")
    parser.add_argument("--url", default=DEFAULT_URL, help="OddsPortal NBA results URL.")
    parser.add_argument(
        "--urls",
        nargs="+",
        help="One or more exact OddsPortal NBA results URLs to fetch.",
    )
    parser.add_argument(
        "--season-start-years",
        nargs="+",
        type=int,
        help="NBA season start years, e.g. 2020 2021 for 2020-2021 and 2021-2022.",
    )
    parser.add_argument(
        "--current-season-start-year",
        type=int,
        help="Use the base /nba/results/ URL for this start year instead of nba-YYYY-YYYY.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--page", type=int, default=1, help="Archive page to fetch first.")
    parser.add_argument(
        "--all-pages",
        action="store_true",
        help="Fetch all archive pages reported by the first page.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Optional cap on pages to fetch. Useful for smoke tests.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refetch pages even when valid page JSON already exists.",
    )
    parser.add_argument(
        "--rebuild-csv-only",
        action="store_true",
        help="Do not fetch anything; rebuild the moneyline CSV from saved page JSON.",
    )
    parser.add_argument("--sqlite-path", type=Path, default=DEFAULT_SQLITE_PATH)
    parser.add_argument(
        "--no-sqlite",
        action="store_true",
        help="Do not rebuild the consolidated SQLite odds database.",
    )
    parser.add_argument(
        "--rebuild-sqlite-only",
        action="store_true",
        help="Do not fetch or rebuild CSVs; rebuild SQLite from existing moneyline CSVs.",
    )
    return parser.parse_args()


def run_url(args: argparse.Namespace, url: str) -> dict[str, Any]:
    season_slug = season_slug_from_url(
        url,
        current_season_start_year=args.current_season_start_year,
    )

    if args.rebuild_csv_only:
        saved_pages = discover_saved_pages(args.output_dir, season_slug)
        moneyline_rows = write_moneyline_csv(args.output_dir, season_slug, saved_pages)
        return {
            "url": url,
            "season_slug": season_slug,
            "pages_loaded": len(saved_pages),
            "moneyline_rows": moneyline_rows,
            "output_dir": str(args.output_dir),
            "mode": "rebuild_csv_only",
        }

    if args.all_pages and not args.force and args.max_pages is None:
        is_complete, page_count, saved_pages = cached_all_pages_available(args.output_dir, season_slug)
        if is_complete:
            moneyline_rows = write_moneyline_csv(args.output_dir, season_slug, saved_pages)
            return {
                "url": url,
                "season_slug": season_slug,
                "pages_fetched": 0,
                "pages_skipped_existing": page_count,
                "pages_saved_valid": len(saved_pages),
                "pages_written_to_csv": page_count,
                "failed_pages": [],
                "reported_page_count": page_count,
                "moneyline_rows": moneyline_rows,
                "output_dir": str(args.output_dir),
                "mode": "cached_all_pages",
            }

    opener = make_opener()
    page_html = fetch_text_with_retries(
        opener,
        url,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        retry_base_seconds=args.retry_base_seconds,
    )
    odds_request = extract_odds_request(page_html)
    user_data_url = extract_user_data_url(page_html, url)
    page_var = extract_page_var(
        fetch_text_with_retries(
            opener,
            user_data_url,
            referer=url,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            retry_base_seconds=args.retry_base_seconds,
        )
    )

    first_payload = None
    fetched_pages = 0
    skipped_pages = 0
    if not args.force:
        first_payload = load_valid_page(args.output_dir, season_slug, args.page)
        if first_payload is not None:
            skipped_pages += 1
            print(f"page {args.page}: using existing {page_path(args.output_dir, season_slug, args.page)}")

    if first_payload is None:
        print(f"page {args.page}: fetching")
        first_payload, odds_request = fetch_archive_page_with_timezone_fallback(
            opener,
            url,
            odds_request,
            page_var,
            args.page,
            args,
        )
        atomic_write_json(page_path(args.output_dir, season_slug, args.page), first_payload)
        fetched_pages += 1

    page_count = first_payload["d"].get("pagination", {}).get("pageCount", 1)
    if args.max_pages is not None:
        page_count = min(page_count, args.max_pages)

    target_pages = [args.page]
    if args.all_pages:
        target_pages = list(range(1, page_count + 1))

    failed_pages = []
    for page in target_pages:
        if page == args.page:
            continue
        existing = None if args.force else load_valid_page(args.output_dir, season_slug, page)
        if existing is not None:
            skipped_pages += 1
            print(f"page {page}/{page_count}: already exists")
            continue

        time.sleep(args.sleep_seconds)
        try:
            print(f"page {page}/{page_count}: fetching")
            payload = fetch_archive_page(
                opener,
                url,
                odds_request,
                page_var,
                page,
                args,
            )
            atomic_write_json(page_path(args.output_dir, season_slug, page), payload)
            fetched_pages += 1
        except Exception as error:
            failed_pages.append({"page": page, "error": str(error)})
            print(f"page {page}/{page_count}: failed: {error}")

    saved_pages = discover_saved_pages(args.output_dir, season_slug)
    csv_pages = filter_pages(saved_pages, set(target_pages) if args.max_pages else None)
    moneyline_rows = write_moneyline_csv(args.output_dir, season_slug, csv_pages)

    summary = {
        "url": url,
        "season_slug": season_slug,
        "pages_fetched": fetched_pages,
        "pages_skipped_existing": skipped_pages,
        "pages_saved_valid": len(saved_pages),
        "pages_written_to_csv": len(csv_pages),
        "failed_pages": failed_pages,
        "reported_page_count": page_count,
        "moneyline_rows": moneyline_rows,
        "output_dir": str(args.output_dir),
    }
    return summary


def main() -> None:
    args = parse_args()
    summaries = []
    if not args.rebuild_sqlite_only:
        for index, url in enumerate(target_urls_from_args(args), start=1):
            print(f"=== OddsPortal URL {index}: {url} ===")
            try:
                summaries.append(run_url(args, url))
            except Exception as error:
                summaries.append(
                    {
                        "url": url,
                        "season_slug": season_slug_from_url(
                            url,
                            current_season_start_year=args.current_season_start_year,
                        ),
                        "error": str(error),
                        "output_dir": str(args.output_dir),
                    }
                )
                print(f"URL failed; continuing: {error}")

    sqlite_summary = None
    if not args.no_sqlite:
        sqlite_summary = build_moneyline_sqlite(args.output_dir, args.sqlite_path)
    print(json.dumps({"runs": summaries, "sqlite": sqlite_summary}, indent=2))


if __name__ == "__main__":
    main()
