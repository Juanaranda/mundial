"""Ajuste por estado actual de los jugadores ("forma").

El modelo Dixon-Coles captura la fuerza histórica de cada selección, pero no
sabe que un crack llega lesionado o que un delantero está on fire en su club.
Este módulo traduce la forma reciente de los jugadores convocados en un ajuste
sobre los parámetros de ataque/defensa de cada selección.

    ataque_ajustado_i  = ataque_i  + BETA_ATK * forma_ofensiva_i
    defensa_ajustada_i = defensa_i + BETA_DEF * forma_defensiva_i

`forma_*` es la desviación (en z-score) del rating reciente de los jugadores
respecto a su nivel base, separando ofensivos (DEL/MED) de defensivos (DEF/GK).

==========================================================================
DE DÓNDE SACAR LA FORMA DE LOS JUGADORES (poblar data/player_form.csv):
  * API-Football (api-sports.io)  -> /players con rating por partido de club
  * SofaScore / FotMob            -> rating 0-10 por partido (no oficial)
  * Transfermarkt                 -> minutos, lesiones, valor de mercado
Formato esperado de data/player_form.csv:
    team,player,position,recent_rating,baseline_rating,available
    Argentina,Lionel Messi,FWD,8.4,7.9,1
    ...
==========================================================================
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

OFFENSIVE = {"FWD", "MID", "DEL", "MED"}
DEFENSIVE = {"DEF", "GK", "DFC", "POR"}

BETA_ATK = 0.18  # cuánto mueve la forma al ataque (calibrar con backtesting)
BETA_DEF = 0.18


def _team_form(group: pd.DataFrame) -> tuple[float, float]:
    available = group[group["available"].astype(int) == 1]
    if available.empty:
        return 0.0, 0.0
    available = available.copy()
    available["delta"] = (available["recent_rating"]
                          - available["baseline_rating"])
    off = available[available["position"].str.upper().isin(OFFENSIVE)]["delta"]
    deff = available[available["position"].str.upper().isin(DEFENSIVE)]["delta"]
    # z-score suave respecto a una desviación típica de rating ~0.6
    f_off = off.mean() / 0.6 if len(off) else 0.0
    f_def = deff.mean() / 0.6 if len(deff) else 0.0
    return float(np.nan_to_num(f_off)), float(np.nan_to_num(f_def))


def load_form_adjustments(path: str | Path | None = None) -> dict[str, dict]:
    """Devuelve {selección: {"atk": x, "def": y}} a partir del CSV de forma.

    Si el archivo no existe, devuelve {} (el modelo corre igual, sin ajuste).
    """
    path = Path(path) if path else DATA_DIR / "player_form.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    adj: dict[str, dict] = {}
    for team, group in df.groupby("team"):
        f_off, f_def = _team_form(group)
        adj[team] = {"atk": BETA_ATK * f_off, "def": BETA_DEF * f_def}
    return adj


def apply_to_model(model, adjustments: dict[str, dict]) -> None:
    """Suma los ajustes de forma a los parámetros del modelo (in-place)."""
    for team, a in adjustments.items():
        if team in model.attack:
            model.attack[team] += a.get("atk", 0.0)
            model.defense[team] += a.get("def", 0.0)


if __name__ == "__main__":
    adj = load_form_adjustments()
    if not adj:
        print("No hay data/player_form.csv todavía — el ajuste por forma se "
              "omite. Mirá el docstring del módulo para saber cómo poblarlo.")
    else:
        for team, a in sorted(adj.items()):
            print(f"{team:<20} atk{a['atk']:+.3f}  def{a['def']:+.3f}")
