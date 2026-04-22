#!/usr/bin/env python3
"""CME-v2 roster counterfactual: add a superteam to Lakers March 2026.

Loads an existing CME-v2 March counterfactual checkpoint and evaluates two OOS
scenarios on Lakers games in March 2026:

  * superteam_added: append all listed players to the Lakers lineup
  * superteam_replace_last4: replace the Lakers' last N lineup slots

The "replace last N" rule is deterministic. Lineups are already ordered by
recency-decayed involvement, so the last slots are the lowest-involvement
players in the constructed roster, not a random draw.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
V2_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(V2_SCRIPTS))

from cme_v2_common import (  # noqa: E402
    BOX_TARGETS,
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_CORE_DB,
    DEFAULT_FEATURES_DB,
    DEFAULT_INJURY_DB,
    DEFAULT_LINEUP_DECAY,
    DEFAULT_LINEUP_LOOKBACK_GAMES,
    DEFAULT_MATCHUP_DB,
    DEFAULT_PLAYER_FORM_DECAY,
    DEFAULT_PLAYER_FORM_LOOKBACK,
    GameDatasetV2,
    PlayerFormStats,
    build_player_form,
    build_records_v2,
    collate_v2,
    fit_tabular_stats,
    load_game_player_status,
    load_game_scores,
    load_games,
    load_matchup_rows_v2,
    load_player_game_stats,
    load_player_histories,
    load_status_calibration,
    load_team_exposures,
    transform_player_form,
)
from cme_v2_model import CmeV2, CmeV2Config  # noqa: E402


DEFAULT_ART = REPO_ROOT / "models_cme_v2" / "artifacts"
DEFAULT_CHECKPOINT = DEFAULT_ART / "multi_player_cf_2026_03_pair" / "model.pt"
DEFAULT_PLAYERS = (
    "giannis:Giannis:203507,"
    "jokic:Jokic:203999,"
    "wemby:Wembanyama:1641705,"
    "shai:Shai:1628983"
)
LAKERS_ID = "1610612747"
WINDOW_START = "2026-03-01"
WINDOW_END = "2026-04-01"


def chrono_train_val(df: pd.DataFrame, val_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    n_val = max(1, int(len(df) * val_frac))
    n_train = len(df) - n_val
    return df.iloc[:n_train].reset_index(drop=True), df.iloc[n_train:].reset_index(drop=True)


@torch.no_grad()
def collect_predictions(
    model: CmeV2,
    records: list,
    *,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    loader = DataLoader(GameDatasetV2(records), batch_size=batch_size,
                        shuffle=False, collate_fn=collate_v2)
    rows: list[pd.DataFrame] = []
    offset = 0
    model.eval()
    for batch in loader:
        batch_dev = {k: v.to(device) for k, v in batch.items()}
        out = model(batch_dev)
        bs = batch["label"].size(0)
        probs = torch.sigmoid(out["win_logit"]).detach().cpu().numpy()
        home_box = out["home_team_box"].detach().cpu().numpy()
        away_box = out["away_team_box"].detach().cpu().numpy()
        margin = out["margin_mu"].detach().cpu().numpy()
        sub_records = records[offset : offset + bs]
        offset += bs

        data = {
            "game_id": [r.game_id for r in sub_records],
            "game_date": [pd.Timestamp(r.game_date).strftime("%Y-%m-%d") for r in sub_records],
            "home_team_id": [r.home_team_id for r in sub_records],
            "away_team_id": [r.away_team_id for r in sub_records],
            "label_home_win": batch["label"].detach().cpu().numpy().astype(int),
            "pred_home_win_prob": probs.astype(float),
            "margin_mu": margin.astype(float),
        }
        for i, stat in enumerate(BOX_TARGETS):
            data[f"pred_home_{stat}"] = home_box[:, i].astype(float)
            data[f"pred_away_{stat}"] = away_box[:, i].astype(float)
        rows.append(pd.DataFrame(data))
    return pd.concat(rows, ignore_index=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--features-db", type=Path, default=DEFAULT_FEATURES_DB)
    p.add_argument("--core-db", type=Path, default=DEFAULT_CORE_DB)
    p.add_argument("--injury-db", type=Path, default=DEFAULT_INJURY_DB)
    p.add_argument("--matchup-db", type=Path, default=DEFAULT_MATCHUP_DB)
    p.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    p.add_argument("--output-root", type=Path, default=DEFAULT_ART)
    p.add_argument("--run-name", type=str, default="superteam_roster_cf_2026_03")
    p.add_argument(
        "--players",
        type=str,
        default=DEFAULT_PLAYERS,
        help="Comma-separated slug:label:player_id entries.",
    )
    p.add_argument("--scenario-name", type=str, default="superteam")
    p.add_argument("--play-prob", type=float, default=1.0)
    p.add_argument("--min-games-before", type=int, default=10)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--lookback-games", type=int, default=DEFAULT_LINEUP_LOOKBACK_GAMES)
    p.add_argument("--decay", type=float, default=DEFAULT_LINEUP_DECAY)
    p.add_argument("--player-form-lookback", type=int, default=DEFAULT_PLAYER_FORM_LOOKBACK)
    p.add_argument("--player-form-decay", type=float, default=DEFAULT_PLAYER_FORM_DECAY)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def parse_players(spec: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        parts = [p.strip() for p in item.split(":")]
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"Bad --players entry {item!r}; expected slug:label:player_id")
        slug, label, player_id = parts
        out.append({"slug": slug, "label": label, "player_id": player_id})
    if not out:
        raise ValueError("--players cannot be empty")
    return out


def player_stats(
    *,
    player_id: str,
    game_date: pd.Timestamp,
    player_histories,
    player_form_stats: PlayerFormStats | None,
    lookback_games: int,
    decay: float,
) -> tuple[float, ...]:
    if player_form_stats is None or player_histories is None:
        return ()
    form = build_player_form(
        player_histories,
        player_id=player_id,
        game_date=game_date,
        lookback_games=lookback_games,
        decay=decay,
    )
    return tuple(transform_player_form(form, player_form_stats).tolist())


def modify_lakers_record(
    record,
    *,
    lakers_home: bool,
    additions: list[dict],
    mode: str,
):
    if lakers_home:
        idx = list(record.home_player_idx)
        prob = list(record.home_play_prob)
        stats = list(record.home_player_stats)
        if mode == "add":
            for item in additions:
                idx.append(item["idx"])
                prob.append(item["prob"])
                stats.append(item["stats"])
        elif mode == "replace_last":
            n = min(len(additions), len(idx))
            start = len(idx) - n
            for offset, item in enumerate(additions[-n:]):
                pos = start + offset
                idx[pos] = item["idx"]
                prob[pos] = item["prob"]
                stats[pos] = item["stats"]
        else:
            raise ValueError(mode)
        return replace(
            record,
            home_player_idx=tuple(idx),
            home_play_prob=tuple(prob),
            home_player_stats=tuple(stats),
        )

    idx = list(record.away_player_idx)
    prob = list(record.away_play_prob)
    stats = list(record.away_player_stats)
    if mode == "add":
        for item in additions:
            idx.append(item["idx"])
            prob.append(item["prob"])
            stats.append(item["stats"])
    elif mode == "replace_last":
        n = min(len(additions), len(idx))
        start = len(idx) - n
        for offset, item in enumerate(additions[-n:]):
            pos = start + offset
            idx[pos] = item["idx"]
            prob[pos] = item["prob"]
            stats[pos] = item["stats"]
    else:
        raise ValueError(mode)
    return replace(
        record,
        away_player_idx=tuple(idx),
        away_play_prob=tuple(prob),
        away_player_stats=tuple(stats),
    )


def make_scenario_records(
    records: list,
    *,
    players: list[dict],
    player_prob: float,
    player_histories,
    player_form_stats,
    args: argparse.Namespace,
    mode: str,
) -> list:
    out = []
    for record in records:
        lakers_home = record.home_team_id == LAKERS_ID
        additions = []
        for player in players:
            stats = player_stats(
                player_id=player["player_id"],
                game_date=pd.Timestamp(record.game_date),
                player_histories=player_histories,
                player_form_stats=player_form_stats,
                lookback_games=args.player_form_lookback,
                decay=args.player_form_decay,
            )
            additions.append({
                "idx": player["idx"],
                "prob": player_prob,
                "stats": stats,
            })
        out.append(
            modify_lakers_record(
                record,
                lakers_home=lakers_home,
                additions=additions,
                mode=mode,
            )
        )
    return out


def add_lakers_columns(df: pd.DataFrame, meta: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    out = df.copy()
    out["game_id"] = out["game_id"].astype(str)
    out = out.merge(meta, on="game_id", how="left")
    out["p_lakers"] = np.where(out["lakers_home"] == 1, out[prob_col], 1.0 - out[prob_col])
    out["pred_lakers_pts"] = np.where(
        out["lakers_home"] == 1, out["pred_home_pts"], out["pred_away_pts"]
    )
    return out


def plot_results(df: pd.DataFrame, out_path: Path, scenario_label: str) -> None:
    df = df.sort_values("game_date").reset_index(drop=True)
    labels = [
        f"{pd.Timestamp(d).strftime('%m-%d')} {'vs' if h == 1 else '@'} {opp}"
        for d, h, opp in zip(df["game_date"], df["lakers_home"], df["opponent_abbr"])
    ]
    x = np.arange(len(df))
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 9), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )

    scenarios = [
        ("baseline", "Baseline", "#1f77b4"),
        ("superteam_added", f"{scenario_label} added", "#2ca02c"),
        ("superteam_replace_last4", f"{scenario_label} replaces last 4", "#ff7f0e"),
    ]
    width = 0.8 / len(scenarios)
    for i, (key, label, color) in enumerate(scenarios):
        col = "p_lakers_baseline" if key == "baseline" else f"p_lakers_{key}"
        ax1.bar(x + (i - 1) * width, df[col], width, label=label, color=color)
    ax1.axhline(0.5, linestyle="--", color="gray", linewidth=0.8, alpha=0.7)
    ax1.set_ylabel("P(Lakers win)")
    ax1.set_ylim(0, 1)
    ax1.set_title(f"CME-v2 Lakers March 2026 roster counterfactual: add {scenario_label}")
    ax1.legend(loc="lower right", ncol=3, framealpha=0.95)
    ax1.grid(axis="y", alpha=0.3)

    delta_scenarios = [
        ("superteam_added", f"{scenario_label} added", "#2ca02c"),
        ("superteam_replace_last4", f"{scenario_label} replaces last 4", "#ff7f0e"),
    ]
    width2 = 0.8 / len(delta_scenarios)
    for i, (key, label, color) in enumerate(delta_scenarios):
        ax2.bar(
            x + (i - 0.5) * width2,
            df[f"delta_{key}"],
            width2,
            label=f"{label}: prob delta",
            color=color,
            alpha=0.85,
        )
    ax2.axhline(0.0, color="black", linewidth=0.6)
    ax2.set_ylabel("Delta P(Lakers win)\n(scenario - baseline)")
    summary = [
        f"added mean={df['delta_superteam_added'].mean():+.3f}, pts={df['delta_lakers_pts_superteam_added'].mean():+.1f}",
        f"replace mean={df['delta_superteam_replace_last4'].mean():+.3f}, pts={df['delta_lakers_pts_superteam_replace_last4'].mean():+.1f}",
    ]
    ax2.set_title(" | ".join(summary), fontsize=10)
    ax2.legend(loc="upper right", framealpha=0.95)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = args.output_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    players = parse_players(args.players)

    print(f"[device] {args.device}")
    print(f"[ckpt] {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = CmeV2Config(**state["cfg"])
    vocab = state["vocab"]
    for player in players:
        player["idx"] = vocab.get(str(player["player_id"]), 0)
    missing = [p for p in players if p["idx"] == 0]
    if missing:
        raise ValueError(f"Players OOV for checkpoint vocab: {missing}")
    scenario_label = " + ".join(p["label"] for p in players)
    replace_scenario = f"{args.scenario_name}_replace_last{len(players)}"
    add_scenario = f"{args.scenario_name}_added"
    team_vocab = state["team_vocab"]
    player_form_stats = None
    if state.get("player_form_means") is not None:
        player_form_stats = PlayerFormStats(
            means=np.array(state["player_form_means"], dtype=np.float64),
            stds=np.array(state["player_form_stds"], dtype=np.float64),
        )

    print("[load] games + histories")
    games_all = load_games(args.features_db, min_games_before=args.min_games_before)
    histories = load_team_exposures(args.core_db)
    game_ids = [str(g) for g in games_all["game_id"].tolist()]
    scores = load_game_scores(args.core_db, game_ids)
    statuses = load_game_player_status(args.injury_db, game_ids)
    calibration = load_status_calibration(args.calibration)
    player_histories = load_player_histories(args.core_db) if player_form_stats is not None else None

    window_start = pd.Timestamp(WINDOW_START)
    window_end = pd.Timestamp(WINDOW_END)
    full_train = games_all[games_all["game_date"] < window_start].copy()
    train_df, _val_df = chrono_train_val(full_train, args.val_frac)
    test_all = games_all[
        (games_all["game_date"] >= window_start)
        & (games_all["game_date"] < window_end)
    ].copy()
    test_df = test_all[
        (test_all["home_team_id"].astype(str) == LAKERS_ID)
        | (test_all["away_team_id"].astype(str) == LAKERS_ID)
    ].copy().reset_index(drop=True)
    print(f"[window] train_fit={len(train_df)} test_lakers={len(test_df)}")

    test_gids = [str(g) for g in test_df["game_id"].tolist()]
    test_matchup = load_matchup_rows_v2(args.matchup_db, test_gids)
    test_pl = load_player_game_stats(args.core_db, test_gids)
    tabular_stats = fit_tabular_stats(train_df)

    baseline_records = build_records_v2(
        test_df,
        histories=histories,
        vocab=type("VocabShim", (), {"encode": lambda _self, p: vocab.get(str(p), 0)})(),
        team_vocab=type("TeamVocabShim", (), {"encode": lambda _self, t: team_vocab.get(str(t), 0)})(),
        status_lookup=statuses,
        calibration=calibration,
        game_scores=scores,
        matchup_rows=test_matchup,
        player_game_stats=test_pl,
        lookback_games=args.lookback_games,
        decay=args.decay,
        tabular_stats=tabular_stats,
        player_histories=player_histories,
        player_form_stats=player_form_stats,
        player_form_lookback=args.player_form_lookback,
        player_form_decay=args.player_form_decay,
        game_odds=None,
    )
    added_records = make_scenario_records(
        baseline_records,
        players=players,
        player_prob=args.play_prob,
        player_histories=player_histories,
        player_form_stats=player_form_stats,
        args=args,
        mode="add",
    )
    replace_records = make_scenario_records(
        baseline_records,
        players=players,
        player_prob=args.play_prob,
        player_histories=player_histories,
        player_form_stats=player_form_stats,
        args=args,
        mode="replace_last",
    )

    model = CmeV2(cfg).to(args.device)
    model.load_state_dict(state["model_state"])
    model.eval()

    print("[predict] baseline / added / replace")
    baseline = collect_predictions(model, baseline_records, batch_size=args.batch_size, device=args.device)
    added = collect_predictions(model, added_records, batch_size=args.batch_size, device=args.device)
    repl = collect_predictions(model, replace_records, batch_size=args.batch_size, device=args.device)

    with sqlite3.connect(args.core_db) as conn:
        abbrev_df = pd.read_sql_query(
            "SELECT game_id, home_team_abbr, away_team_abbr FROM games", conn
        )
    abbrev_df["game_id"] = abbrev_df["game_id"].astype(str)
    meta = test_df[["game_id", "home_team_id", "away_team_id"]].copy()
    meta["game_id"] = meta["game_id"].astype(str)
    meta = meta.merge(abbrev_df, on="game_id", how="left")
    meta["lakers_home"] = (meta["home_team_id"].astype(str) == LAKERS_ID).astype(int)
    meta["opponent_abbr"] = np.where(
        meta["lakers_home"] == 1, meta["away_team_abbr"], meta["home_team_abbr"]
    )
    meta = meta[["game_id", "lakers_home", "opponent_abbr"]]

    b = add_lakers_columns(baseline, meta, "pred_home_win_prob")
    a = add_lakers_columns(added, meta, "pred_home_win_prob")
    r = add_lakers_columns(repl, meta, "pred_home_win_prob")

    merged = b[[
        "game_id", "game_date", "lakers_home", "opponent_abbr",
        "label_home_win", "pred_home_win_prob", "p_lakers", "pred_lakers_pts",
    ]].rename(columns={
        "pred_home_win_prob": "p_home_baseline",
        "p_lakers": "p_lakers_baseline",
        "pred_lakers_pts": "pred_lakers_pts_baseline",
    })
    for name, frame in ((add_scenario, a), (replace_scenario, r)):
        sub = frame[["game_id", "pred_home_win_prob", "p_lakers", "pred_lakers_pts"]].rename(
            columns={
                "pred_home_win_prob": f"p_home_{name}",
                "p_lakers": f"p_lakers_{name}",
                "pred_lakers_pts": f"pred_lakers_pts_{name}",
            }
        )
        merged = merged.merge(sub, on="game_id", how="left")
        merged[f"delta_{name}"] = merged[f"p_lakers_{name}"] - merged["p_lakers_baseline"]
        merged[f"delta_lakers_pts_{name}"] = (
            merged[f"pred_lakers_pts_{name}"] - merged["pred_lakers_pts_baseline"]
        )

    csv_path = out_dir / "lakers_march_superteam_roster.csv"
    merged.to_csv(csv_path, index=False)
    png_path = out_dir / "lakers_march_superteam_roster.png"
    plot_results(merged, png_path, scenario_label)

    summary = {
        "checkpoint": str(args.checkpoint),
        "players": [
            {
                "slug": p["slug"],
                "label": p["label"],
                "player_id": p["player_id"],
                "vocab_idx": int(p["idx"]),
            }
            for p in players
        ],
        "play_prob": args.play_prob,
        "n_test_lakers": int(len(merged)),
        "mean_p_lakers_baseline": float(merged["p_lakers_baseline"].mean()),
        "mean_lakers_pts_baseline": float(merged["pred_lakers_pts_baseline"].mean()),
        "scenarios": {
            add_scenario: {
                "mean_delta": float(merged[f"delta_{add_scenario}"].mean()),
                "median_delta": float(merged[f"delta_{add_scenario}"].median()),
                "min_delta": float(merged[f"delta_{add_scenario}"].min()),
                "max_delta": float(merged[f"delta_{add_scenario}"].max()),
                "mean_delta_lakers_pts": float(merged[f"delta_lakers_pts_{add_scenario}"].mean()),
            },
            replace_scenario: {
                "mean_delta": float(merged[f"delta_{replace_scenario}"].mean()),
                "median_delta": float(merged[f"delta_{replace_scenario}"].median()),
                "min_delta": float(merged[f"delta_{replace_scenario}"].min()),
                "max_delta": float(merged[f"delta_{replace_scenario}"].max()),
                "mean_delta_lakers_pts": float(merged[f"delta_lakers_pts_{replace_scenario}"].mean()),
            },
        },
        "cfg": asdict(cfg),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[save] {csv_path}")
    print(f"[save] {png_path}")
    print(f"[save] {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
