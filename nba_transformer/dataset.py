"""Self-contained data layer for the NBA-Transformer.

Reads from the precomputed feature SQLite database produced by
`scripts/build_features_db.py`. The runtime path here has no dependencies on
any prior feature-building modules — it just loads tensors that have already
been materialized as blobs.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import torch
from torch.utils.data import Dataset


N_SEGMENTS = 48  # per-minute rotation horizon
K_BOX = 14       # box-score targets per player-minute
K_PAIR = 9       # off-vs-def pair supervision targets


def collate(batch: list[dict]) -> dict:
    """Pad rosters to the max length in the batch, concat flat sup_* tensors,
    pad rotation targets to N_SEGMENTS, and stack scalar fields.
    """
    B = len(batch)
    L_h = max(b["home_idx"].numel() for b in batch)
    L_a = max(b["away_idx"].numel() for b in batch)
    D_stats = batch[0]["home_stats"].size(-1)

    home_idx = torch.zeros(B, L_h, dtype=torch.long)
    home_mask = torch.zeros(B, L_h, dtype=torch.bool)
    away_idx = torch.zeros(B, L_a, dtype=torch.long)
    away_mask = torch.zeros(B, L_a, dtype=torch.bool)
    home_career_year_idx = torch.zeros(B, L_h, dtype=torch.long)
    away_career_year_idx = torch.zeros(B, L_a, dtype=torch.long)
    home_stats = torch.zeros(B, L_h, D_stats, dtype=torch.float32)
    away_stats = torch.zeros(B, L_a, D_stats, dtype=torch.float32)
    home_minutes = torch.zeros(B, L_h, dtype=torch.float32)
    away_minutes = torch.zeros(B, L_a, dtype=torch.float32)
    home_rotation = torch.zeros(B, L_h, N_SEGMENTS, dtype=torch.float32)
    away_rotation = torch.zeros(B, L_a, N_SEGMENTS, dtype=torch.float32)
    has_minute_box = "home_minute_box" in batch[0]
    if has_minute_box:
        home_minute_box = torch.zeros(B, L_h, N_SEGMENTS, K_BOX, dtype=torch.float32)
        away_minute_box = torch.zeros(B, L_a, N_SEGMENTS, K_BOX, dtype=torch.float32)

    for b, item in enumerate(batch):
        n_h = item["home_idx"].numel()
        n_a = item["away_idx"].numel()
        home_idx[b, :n_h] = item["home_idx"]
        home_mask[b, :n_h] = True
        away_idx[b, :n_a] = item["away_idx"]
        away_mask[b, :n_a] = True
        home_career_year_idx[b, :n_h] = item["home_career_year_idx"]
        away_career_year_idx[b, :n_a] = item["away_career_year_idx"]
        if D_stats > 0:
            home_stats[b, :n_h] = item["home_stats"]
            away_stats[b, :n_a] = item["away_stats"]
        home_minutes[b, :n_h] = item["home_minutes_actual"]
        away_minutes[b, :n_a] = item["away_minutes_actual"]
        home_rotation[b, :n_h] = item["home_rotation_target"]
        away_rotation[b, :n_a] = item["away_rotation_target"]
        if has_minute_box:
            n_hm = item["home_minute_box"].size(0)
            n_am = item["away_minute_box"].size(0)
            home_minute_box[b, :n_hm] = item["home_minute_box"]
            away_minute_box[b, :n_am] = item["away_minute_box"]

    out = {
        "home_idx": home_idx,
        "home_mask": home_mask,
        "away_idx": away_idx,
        "away_mask": away_mask,
        "home_career_year_idx": home_career_year_idx,
        "away_career_year_idx": away_career_year_idx,
        "home_stats": home_stats,
        "away_stats": away_stats,
        "home_team_idx": torch.stack([b["home_team_idx"] for b in batch]),
        "away_team_idx": torch.stack([b["away_team_idx"] for b in batch]),
        "home_rest": torch.stack([b["home_rest"] for b in batch]),
        "away_rest": torch.stack([b["away_rest"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "margin": torch.stack([b["margin"] for b in batch]),
        "tabular": torch.stack([b["tabular"] for b in batch]),
        "home_dec_odds": torch.stack([b["home_dec_odds"] for b in batch]),
        "away_dec_odds": torch.stack([b["away_dec_odds"] for b in batch]),
        "has_odds": torch.stack([b["has_odds"] for b in batch]),
        "team_box_home": torch.stack([b["team_box_home"] for b in batch]),
        "team_box_away": torch.stack([b["team_box_away"] for b in batch]),
        "home_minutes_actual": home_minutes,
        "away_minutes_actual": away_minutes,
        "home_pace_actual": torch.stack([b["home_pace_actual"] for b in batch]),
        "away_pace_actual": torch.stack([b["away_pace_actual"] for b in batch]),
        "home_minutes_valid": torch.stack([b["home_minutes_valid"] for b in batch]),
        "away_minutes_valid": torch.stack([b["away_minutes_valid"] for b in batch]),
        "home_rotation_target": home_rotation,
        "away_rotation_target": away_rotation,
        "home_rotation_valid": torch.stack([b["home_rotation_valid"] for b in batch]),
        "away_rotation_valid": torch.stack([b["away_rotation_valid"] for b in batch]),
    }
    if has_minute_box:
        out["home_minute_box"] = home_minute_box
        out["away_minute_box"] = away_minute_box

    pair_game = torch.cat([
        torch.full_like(b["sup_pair_side"], i) for i, b in enumerate(batch)
    ]) if any(b["sup_pair_side"].numel() for b in batch) else torch.zeros(0, dtype=torch.long)
    out["sup_pair_game"] = pair_game
    out["sup_pair_side"] = torch.cat([b["sup_pair_side"] for b in batch])
    out["sup_pair_off"] = torch.cat([b["sup_pair_off"] for b in batch])
    out["sup_pair_def"] = torch.cat([b["sup_pair_def"] for b in batch])
    out["sup_pair_y"] = torch.cat([b["sup_pair_y"] for b in batch], dim=0) \
        if any(b["sup_pair_y"].numel() for b in batch) \
        else torch.zeros(0, K_PAIR, dtype=torch.float32)

    pl_game = torch.cat([
        torch.full_like(b["sup_pl_side"], i) for i, b in enumerate(batch)
    ]) if any(b["sup_pl_side"].numel() for b in batch) else torch.zeros(0, dtype=torch.long)
    out["sup_pl_game"] = pl_game
    out["sup_pl_side"] = torch.cat([b["sup_pl_side"] for b in batch])
    out["sup_pl_slot"] = torch.cat([b["sup_pl_slot"] for b in batch])
    out["sup_pl_y"] = torch.cat([b["sup_pl_y"] for b in batch], dim=0) \
        if any(b["sup_pl_y"].numel() for b in batch) \
        else torch.zeros(0, K_BOX, dtype=torch.float32)

    return out


class PrecomputedDataset(Dataset):
    """Loads pre-computed features from SQLite (built by build_features_db.py).

    All blobs are decoded once in __init__ so __getitem__ is a pure-Python
    materialization with no DB or disk access.
    """

    def __init__(self, db_path: str | Path, window_start: str, split: str,
                 regular_season_only: bool = False) -> None:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # Regular-season-only filter: NBA game IDs encode game type in positions 4-5
        # ("00" = regular season, "01" = playoffs).
        if regular_season_only:
            game_rows = conn.execute(
                "SELECT * FROM games WHERE window_start=? AND split=? "
                "AND substr(game_id, 4, 2)='00'",
                (window_start, split),
            ).fetchall()
        else:
            game_rows = conn.execute(
                "SELECT * FROM games WHERE window_start=? AND split=?",
                (window_start, split),
            ).fetchall()
        self._game_ids = [row["game_id"] for row in game_rows]
        self._games: dict[str, dict] = {}
        for row in game_rows:
            self._games[row["game_id"]] = {
                **{k: row[k] for k in ("game_id", "label", "margin",
                    "home_team_idx", "away_team_idx", "home_rest", "away_rest",
                    "home_dec_odds", "away_dec_odds", "has_odds",
                    "home_pace_actual", "away_pace_actual",
                    "home_minutes_valid", "away_minutes_valid",
                    "home_rotation_valid", "away_rotation_valid")},
                "tabular": torch.frombuffer(bytearray(row["tabular"]), dtype=torch.float32).clone(),
                "team_box_home": torch.frombuffer(bytearray(row["team_box_home"]), dtype=torch.float32).clone(),
                "team_box_away": torch.frombuffer(bytearray(row["team_box_away"]), dtype=torch.float32).clone(),
            }

        gid_placeholders = ",".join("?" * len(self._game_ids))
        player_rows = conn.execute(
            f"SELECT * FROM players WHERE window_start=? AND game_id IN ({gid_placeholders})",
            [window_start] + self._game_ids,
        ).fetchall()
        self._players: dict[str, list] = {}
        for row in player_rows:
            fs = row["form_stats"]
            rt = row["rotation_target"]
            rec = {
                "side": row["side"], "slot": row["slot"],
                "player_idx": row["player_idx"],
                "career_year_idx": row["career_year_idx"],
                "minutes_actual": row["minutes_actual"],
                "form_stats": torch.frombuffer(bytearray(fs), dtype=torch.float32).clone() if fs else None,
                "rotation_target": torch.frombuffer(bytearray(rt), dtype=torch.float32).clone(),
            }
            self._players.setdefault(row["game_id"], []).append(rec)

        label_rows = conn.execute(
            f"SELECT * FROM player_labels WHERE window_start=? AND game_id IN ({gid_placeholders})",
            [window_start] + self._game_ids,
        ).fetchall()
        self._player_labels: dict[str, list] = {}
        for row in label_rows:
            rec = {
                "side": row["side"], "slot": row["slot"],
                "targets": torch.frombuffer(bytearray(row["targets"]), dtype=torch.float32).clone(),
            }
            self._player_labels.setdefault(row["game_id"], []).append(rec)

        pair_rows = conn.execute(
            f"SELECT * FROM pairs WHERE window_start=? AND game_id IN ({gid_placeholders})",
            [window_start] + self._game_ids,
        ).fetchall()
        self._pairs: dict[str, list] = {}
        for row in pair_rows:
            rec = {
                "side": row["side"], "off_slot": row["off_slot"], "def_slot": row["def_slot"],
                "targets": torch.frombuffer(bytearray(row["targets"]), dtype=torch.float32).clone(),
            }
            self._pairs.setdefault(row["game_id"], []).append(rec)

        self._minute_box: dict[str, list] = {}
        try:
            mb_rows = conn.execute(
                f"SELECT * FROM player_minute_box WHERE window_start=? AND game_id IN ({gid_placeholders})",
                [window_start] + self._game_ids,
            ).fetchall()
            for row in mb_rows:
                rec = {
                    "side": row["side"], "slot": row["slot"],
                    "stats": torch.frombuffer(bytearray(row["stats_blob"]), dtype=torch.float32).clone().view(N_SEGMENTS, K_BOX),
                }
                self._minute_box.setdefault(row["game_id"], []).append(rec)
        except sqlite3.OperationalError:
            pass

        conn.close()

    def __len__(self) -> int:
        return len(self._game_ids)

    def __getitem__(self, idx: int) -> dict:
        gid = self._game_ids[idx]
        g = self._games[gid]

        players = sorted(self._players.get(gid, []), key=lambda r: (r["side"], r["slot"]))
        home_players = [p for p in players if p["side"] == 0]
        away_players = [p for p in players if p["side"] == 1]
        L_h = len(home_players)
        L_a = len(away_players)

        home_idx = torch.tensor([p["player_idx"] for p in home_players], dtype=torch.long)
        away_idx = torch.tensor([p["player_idx"] for p in away_players], dtype=torch.long)
        home_career = torch.tensor([p["career_year_idx"] for p in home_players], dtype=torch.long)
        away_career = torch.tensor([p["career_year_idx"] for p in away_players], dtype=torch.long)

        if home_players and home_players[0]["form_stats"] is not None:
            home_stats = torch.stack([p["form_stats"] for p in home_players])
            away_stats = torch.stack([p["form_stats"] for p in away_players])
        else:
            home_stats = torch.zeros(L_h, 0, dtype=torch.float32)
            away_stats = torch.zeros(L_a, 0, dtype=torch.float32)

        home_minutes = torch.tensor([p["minutes_actual"] for p in home_players], dtype=torch.float32)
        away_minutes = torch.tensor([p["minutes_actual"] for p in away_players], dtype=torch.float32)

        home_rotation = torch.stack([p["rotation_target"] for p in home_players]) \
            if home_players else torch.zeros(0, N_SEGMENTS, dtype=torch.float32)
        away_rotation = torch.stack([p["rotation_target"] for p in away_players]) \
            if away_players else torch.zeros(0, N_SEGMENTS, dtype=torch.float32)

        pl_labels = self._player_labels.get(gid, [])
        if pl_labels:
            sup_pl_side = torch.tensor([p["side"] for p in pl_labels], dtype=torch.long)
            sup_pl_slot = torch.tensor([p["slot"] for p in pl_labels], dtype=torch.long)
            sup_pl_y = torch.stack([p["targets"] for p in pl_labels])
        else:
            sup_pl_side = torch.zeros(0, dtype=torch.long)
            sup_pl_slot = torch.zeros(0, dtype=torch.long)
            sup_pl_y = torch.zeros(0, K_BOX, dtype=torch.float32)

        pairs = self._pairs.get(gid, [])
        if pairs:
            sup_pair_side = torch.tensor([p["side"] for p in pairs], dtype=torch.long)
            sup_pair_off = torch.tensor([p["off_slot"] for p in pairs], dtype=torch.long)
            sup_pair_def = torch.tensor([p["def_slot"] for p in pairs], dtype=torch.long)
            sup_pair_y = torch.stack([p["targets"] for p in pairs])
        else:
            sup_pair_side = torch.zeros(0, dtype=torch.long)
            sup_pair_off = torch.zeros(0, dtype=torch.long)
            sup_pair_def = torch.zeros(0, dtype=torch.long)
            sup_pair_y = torch.zeros(0, K_PAIR, dtype=torch.float32)

        mb_rows = sorted(self._minute_box.get(gid, []), key=lambda r: (r["side"], r["slot"]))
        home_mb = [r for r in mb_rows if r["side"] == 0]
        away_mb = [r for r in mb_rows if r["side"] == 1]
        home_minute_box = torch.stack([r["stats"] for r in home_mb]) \
            if home_mb else torch.zeros(L_h, N_SEGMENTS, K_BOX, dtype=torch.float32)
        away_minute_box = torch.stack([r["stats"] for r in away_mb]) \
            if away_mb else torch.zeros(L_a, N_SEGMENTS, K_BOX, dtype=torch.float32)

        return {
            "home_idx": home_idx,
            "away_idx": away_idx,
            "home_career_year_idx": home_career,
            "away_career_year_idx": away_career,
            "home_stats": home_stats,
            "away_stats": away_stats,
            "home_team_idx": torch.tensor(g["home_team_idx"], dtype=torch.long),
            "away_team_idx": torch.tensor(g["away_team_idx"], dtype=torch.long),
            "home_rest": torch.tensor(g["home_rest"], dtype=torch.float32),
            "away_rest": torch.tensor(g["away_rest"], dtype=torch.float32),
            "label": torch.tensor(g["label"], dtype=torch.float32),
            "margin": torch.tensor(g["margin"], dtype=torch.float32),
            "tabular": g["tabular"],
            "home_dec_odds": torch.tensor(g["home_dec_odds"], dtype=torch.float32),
            "away_dec_odds": torch.tensor(g["away_dec_odds"], dtype=torch.float32),
            "has_odds": torch.tensor(float(g["has_odds"]), dtype=torch.float32),
            "team_box_home": g["team_box_home"],
            "team_box_away": g["team_box_away"],
            "sup_pair_side": sup_pair_side,
            "sup_pair_off": sup_pair_off,
            "sup_pair_def": sup_pair_def,
            "sup_pair_y": sup_pair_y,
            "sup_pl_side": sup_pl_side,
            "sup_pl_slot": sup_pl_slot,
            "sup_pl_y": sup_pl_y,
            "home_minutes_actual": home_minutes,
            "away_minutes_actual": away_minutes,
            "home_pace_actual": torch.tensor(g["home_pace_actual"], dtype=torch.float32),
            "away_pace_actual": torch.tensor(g["away_pace_actual"], dtype=torch.float32),
            "home_minutes_valid": torch.tensor(float(g["home_minutes_valid"]), dtype=torch.float32),
            "away_minutes_valid": torch.tensor(float(g["away_minutes_valid"]), dtype=torch.float32),
            "home_rotation_target": home_rotation,
            "away_rotation_target": away_rotation,
            "home_rotation_valid": torch.tensor(float(g["home_rotation_valid"]), dtype=torch.float32),
            "away_rotation_valid": torch.tensor(float(g["away_rotation_valid"]), dtype=torch.float32),
            "home_minute_box": home_minute_box,
            "away_minute_box": away_minute_box,
        }


def load_window_info(db_path: str | Path) -> list[tuple[str, str]]:
    """Return list of (window_start, window_end) available in the precomputed DB."""
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT DISTINCT window_start, window_end FROM games ORDER BY window_start"
    ).fetchall()
    conn.close()
    return [(r[0], r[1]) for r in rows]


def load_vocab_size(db_path: str | Path, window_start: str) -> tuple[int, int]:
    """Return (vocab_size, team_vocab_size) for a given window."""
    conn = sqlite3.connect(str(db_path))
    vs = conn.execute("SELECT value FROM meta WHERE key=?",
                      (f"vocab_size_{window_start}",)).fetchone()
    tvs = conn.execute("SELECT value FROM meta WHERE key=?",
                       (f"team_vocab_size_{window_start}",)).fetchone()
    conn.close()
    return int(vs[0]), int(tvs[0])


__all__ = [
    "N_SEGMENTS", "K_BOX", "K_PAIR",
    "PrecomputedDataset", "collate",
    "load_window_info", "load_vocab_size",
]
