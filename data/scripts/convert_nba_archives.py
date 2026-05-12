#!/usr/bin/env python3
"""Convert selected shufinskiy/nba_data CSV archives into the raw SQLite DB."""

from __future__ import annotations

import argparse
import csv
import sqlite3
import tarfile
from collections.abc import Iterable
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PROJECT_ROOT / "data" / "artifacts"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "external_repos" / "nba_data" / "datasets"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "nba_raw.sqlite"

NBA_SOURCES = ("cdnnba", "nbastatsv3", "shotdetail", "matchups", "pbpstats")


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def archive_path(dataset_dir: Path, source: str, season: int) -> Path:
    return dataset_dir / f"{source}_{season}.tar.xz"


def table_name(source: str, season: int) -> str:
    return f"{source}_{season}"


def iter_csv_rows(path: Path) -> tuple[list[str], Iterable[dict[str, str]]]:
    tar = tarfile.open(path, "r:xz")
    members = [member for member in tar.getmembers() if member.isfile()]
    if len(members) != 1:
        tar.close()
        raise ValueError(f"Expected one CSV member in {path}, found {len(members)}")

    extracted = tar.extractfile(members[0])
    if extracted is None:
        tar.close()
        raise ValueError(f"Could not read CSV member from {path}")

    lines = (line.decode("utf-8-sig", errors="replace") for line in extracted)
    reader = csv.DictReader(lines)
    columns = reader.fieldnames or []

    def rows() -> Iterable[dict[str, str]]:
        try:
            yield from reader
        finally:
            extracted.close()
            tar.close()

    return columns, rows()


def create_table(conn: sqlite3.Connection, name: str, columns: list[str], replace: bool) -> None:
    quoted_name = quote_ident(name)
    if replace:
        conn.execute(f"DROP TABLE IF EXISTS {quoted_name}")

    raw_columns = ", ".join(f"{quote_ident(column)} TEXT" for column in columns)
    metadata_columns = """
        "_source_archive" TEXT NOT NULL,
        "_source_member" TEXT NOT NULL,
        "_loaded_at" TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    """
    conn.execute(f"CREATE TABLE IF NOT EXISTS {quoted_name} ({raw_columns}, {metadata_columns})")


def source_member(path: Path) -> str:
    with tarfile.open(path, "r:xz") as tar:
        members = [member.name for member in tar.getmembers() if member.isfile()]
    if len(members) != 1:
        raise ValueError(f"Expected one CSV member in {path}, found {len(members)}")
    return members[0]


def insert_rows(
    conn: sqlite3.Connection,
    name: str,
    columns: list[str],
    rows: Iterable[dict[str, str]],
    source_archive: Path,
    member: str,
    batch_size: int,
    limit: int | None,
) -> int:
    all_columns = columns + ["_source_archive", "_source_member"]
    placeholders = ", ".join("?" for _ in all_columns)
    insert_sql = (
        f"INSERT INTO {quote_ident(name)} "
        f"({', '.join(quote_ident(column) for column in all_columns)}) "
        f"VALUES ({placeholders})"
    )

    batch: list[tuple[str, ...]] = []
    count = 0
    for row in rows:
        values = tuple(row.get(column, "") for column in columns)
        values += (str(source_archive.relative_to(PROJECT_ROOT)), member)
        batch.append(values)
        count += 1
        if len(batch) >= batch_size:
            conn.executemany(insert_sql, batch)
            batch.clear()
        if limit is not None and count >= limit:
            break

    if batch:
        conn.executemany(insert_sql, batch)
    return count


def add_indexes(conn: sqlite3.Connection, name: str, columns: list[str]) -> None:
    candidates = [
        "gameId",
        "GAME_ID",
        "game_id",
        "teamId",
        "TEAM_ID",
        "team_id",
        "personId",
        "PLAYER_ID",
        "person_id",
        "GAME_DATE",
        "timeActual",
    ]
    for column in candidates:
        if column not in columns:
            continue
        index_name = f"idx_{name}_{column}".replace("-", "_")
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS {quote_ident(index_name)} "
            f"ON {quote_ident(name)} ({quote_ident(column)})"
        )


def convert_archive(
    conn: sqlite3.Connection,
    path: Path,
    name: str,
    replace: bool,
    batch_size: int,
    limit: int | None,
) -> int:
    columns, rows = iter_csv_rows(path)
    if not columns:
        raise ValueError(f"No CSV header found in {path}")

    member = source_member(path)
    create_table(conn, name, columns, replace=replace)
    count = insert_rows(conn, name, columns, rows, path, member, batch_size, limit)
    add_indexes(conn, name, columns)
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream selected nba_data .tar.xz CSV archives into raw SQLite tables."
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        required=True,
        help="Season start years, e.g. 2020 2021 2022 2023 2024 2025.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(NBA_SOURCES),
        choices=NBA_SOURCES,
        help="Archive sources to convert.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing nba_data .tar.xz archives.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="SQLite output path. Defaults to data/artifacts/nba_raw.sqlite.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append into existing tables. Defaults to replacing selected tables.",
    )
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--limit", type=int, help="Optional row limit per source for testing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(args.output) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")

        for season in args.seasons:
            for source in args.sources:
                path = archive_path(args.dataset_dir, source, season)
                if not path.exists():
                    print(f"Skipping missing archive: {path}")
                    continue

                name = table_name(source, season)
                row_count = convert_archive(
                    conn=conn,
                    path=path,
                    name=name,
                    replace=not args.append,
                    batch_size=args.batch_size,
                    limit=args.limit,
                )
                conn.commit()
                print(f"{name}: inserted {row_count} rows from {path}")

    print(f"SQLite database: {args.output}")


if __name__ == "__main__":
    main()
