"""Ratings Elo dinámicos para selecciones.

El Elo refleja mejor la fuerza relativa para predicción que el ranking FIFA
clásico. Se recorre el histórico cronológicamente y se actualiza el rating de
cada selección después de cada partido.

Ajustes sobre el Elo básico (estilo eloratings.net):
  * K depende de la importancia del torneo (amistoso < clasificatoria < Mundial)
  * el margen de goles aumenta el cambio de rating
  * ventaja de localía sumada al rating del local (salvo cancha neutral)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple

import pandas as pd

BASE_RATING = 1500.0
HOME_ADVANTAGE = 65.0  # puntos Elo de ventaja por jugar de local

# Peso del torneo -> factor K
TOURNAMENT_K = {
    "FIFA World Cup": 60,
    "FIFA World Cup qualification": 40,
    "Copa América": 50,
    "UEFA Euro": 50,
    "UEFA Euro qualification": 40,
    "African Cup of Nations": 50,
    "AFC Asian Cup": 50,
    "Confederations Cup": 45,
    "UEFA Nations League": 40,
    "Friendly": 20,
}
DEFAULT_K = 30


def _k_factor(tournament: str) -> float:
    return TOURNAMENT_K.get(tournament, DEFAULT_K)


def _expected(rating_a: float, rating_b: float) -> float:
    """Probabilidad esperada de que A le gane a B."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _goal_multiplier(goal_diff: int) -> float:
    """El Elo se mueve más cuando la goleada es más amplia."""
    g = abs(goal_diff)
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11 + g) / 8.0


def compute_elo(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Calcula el Elo partido a partido.

    Devuelve:
      * el DataFrame con columnas `home_elo` / `away_elo` (rating ANTES del
        partido), útil como feature para el modelo de goles.
      * un dict {selección: rating final}.
    """
    ratings: Dict[str, float] = defaultdict(lambda: BASE_RATING)
    home_elos, away_elos = [], []

    for row in df.itertuples(index=False):
        ra = ratings[row.home_team]
        rb = ratings[row.away_team]
        home_elos.append(ra)
        away_elos.append(rb)

        adv = 0.0 if getattr(row, "neutral", False) else HOME_ADVANTAGE
        exp_home = _expected(ra + adv, rb)

        if row.home_score > row.away_score:
            score_home = 1.0
        elif row.home_score < row.away_score:
            score_home = 0.0
        else:
            score_home = 0.5

        k = _k_factor(row.tournament) * _goal_multiplier(
            row.home_score - row.away_score
        )
        delta = k * (score_home - exp_home)
        ratings[row.home_team] = ra + delta
        ratings[row.away_team] = rb - delta

    out = df.copy()
    out["home_elo"] = home_elos
    out["away_elo"] = away_elos
    return out, dict(ratings)


def win_probabilities(
    rating_home: float, rating_away: float, neutral: bool = True
) -> Tuple[float, float, float]:
    """Probabilidad (victoria local, empate, victoria visita) según Elo.

    El empate se modela con una curva empírica centrada en diferencias de
    rating pequeñas (no hay empate en el Elo puro, así que lo aproximamos).
    """
    adv = 0.0 if neutral else HOME_ADVANTAGE
    e_home = _expected(rating_home + adv, rating_away)
    # P(empate) decae a medida que crece la diferencia de fuerza.
    diff = abs((rating_home + adv) - rating_away)
    p_draw = 0.30 * (2.718 ** (-(diff / 350.0) ** 2))
    p_home = e_home * (1 - p_draw)
    p_away = (1 - e_home) * (1 - p_draw)
    return p_home, p_draw, p_away


if __name__ == "__main__":
    from data import load_results

    df = load_results()
    df, final = compute_elo(df)
    top = sorted(final.items(), key=lambda kv: kv[1], reverse=True)[:20]
    print("Top 20 selecciones por Elo (al último partido del dataset):\n")
    for i, (team, r) in enumerate(top, 1):
        print(f"{i:>2}. {team:<20} {r:7.1f}")
