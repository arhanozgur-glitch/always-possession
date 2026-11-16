"""Reader for Metrica's FIFA/EPTS tracking export (metadata XML + raw text).

This is the format of the competition tracking and of Metrica Sample Game 3, so the same reader
serves both. Raw line: "<frame>:<x,y[,speed]>;<x,y[,speed]>;...:<ball x,y[,...]>", coordinates
normalised 0-1, "NaN" for a missing observation.

Which player each ";"-group belongs to is declared in the metadata: every
<DataFormatSpecification startFrame endFrame> lists the player channels in line order, and the
list can change during the match (substitutions). Track slots here follow the order of the
<Player> entries; the first <Team> is the home side (the organizer baseline's convention).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .data import AWAY, HOME


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def read_metadata(path: Path) -> dict:
    root = ET.parse(path).getroot()
    teams, player_ids, player_team, params, size = [], [], [], {}, {}
    channel_player, specs = {}, []
    fps = 25.0
    for el in root.iter():
        tag = _local(el.tag)
        if tag == "Team" and el.get("id"):
            teams.append(el.get("id"))
        elif tag == "Player" and el.get("teamId"):
            player_ids.append(el.get("id"))
            player_team.append(el.get("teamId"))
        elif tag == "PlayerChannel" and el.get("channelId") == "x":
            channel_player[el.get("id")] = el.get("playerId")
        elif tag == "DataFormatSpecification":
            refs = [r.get("playerChannelId") for r in el.iter() if _local(r.tag) == "PlayerChannelRef"]
            specs.append((int(el.get("startFrame")), int(el.get("endFrame")),
                          [channel_player[r] for r in refs if r in channel_player]))
        elif tag == "FrameRate" and el.text:
            fps = float(el.text)
        elif tag == "ProviderParameter":
            kids = {_local(k.tag): (k.text or "") for k in el}
            if kids.get("Name"):
                params[kids["Name"]] = kids.get("Value", "")
        elif tag == "FieldSize":
            size = {_local(k.tag): float(k.text) for k in el if k.text}
    if len(teams) != 2 or not player_ids:
        raise ValueError(f"{path}: expected 2 teams and some players, got {len(teams)} / {len(player_ids)}")
    slot = {pid: i for i, pid in enumerate(player_ids)}
    team = np.array([HOME if t == teams[0] else AWAY for t in player_team], dtype=np.int8)
    layouts = [(a, b, np.array([slot[p] for p in pids], dtype=np.int64)) for a, b, pids in specs if pids]
    return {"team": team, "team_ids": teams, "fps": fps, "params": params, "layouts": layouts,
            "pitch": (size.get("Width", 105.0), size.get("Height", 68.0))}


def _pair(group: str) -> tuple[float, float]:
    if not group or group[0] in "Nn":
        return np.nan, np.nan
    parts = group.split(",")
    return float(parts[0]), float(parts[1])


def read_raw(path: Path, meta: dict, pitch: tuple[float, float] | None = None,
             n_frames: int | None = None):
    """(xy [T,S,2] metres, ball [T,2] metres). Row i is frame_id i + 1; absent frames stay NaN."""
    n_slots = len(meta["team"])
    layouts = meta["layouts"]
    identity = np.arange(n_slots)
    frames, rows, balls = [], [], []
    k = 0
    with open(path) as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0])
            order = identity
            if layouts:
                while k + 1 < len(layouts) and frame > layouts[k][1]:
                    k += 1
                while k > 0 and frame < layouts[k][0]:
                    k -= 1
                order = layouts[k][2]
            row = np.full((n_slots, 2), np.nan, dtype=np.float32)
            for slot, group in zip(order, parts[1].split(";")):
                row[slot] = _pair(group)
            frames.append(frame)
            rows.append(row)
            balls.append(_pair(parts[2]))
    total = n_frames or max(frames)
    xy = np.full((total, n_slots, 2), np.nan, dtype=np.float32)
    ball = np.full((total, 2), np.nan, dtype=np.float32)
    idx = np.array(frames) - 1
    keep = (idx >= 0) & (idx < total)
    xy[idx[keep]] = np.array(rows, dtype=np.float32)[keep]
    ball[idx[keep]] = np.array(balls, dtype=np.float32)[keep]
    scale = np.array(pitch or meta["pitch"], dtype=np.float32)
    return xy * scale, ball * scale
