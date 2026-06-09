"""Simulación Monte Carlo del torneo.

Con el modelo de marcadores ya ajustado, simulamos el Mundial miles de veces
para estimar la probabilidad de que cada selección avance de fase y sea campeón.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List

import numpy as np


def simulate_match(model, home, away, knockout=False):
    """Simula un partido y devuelve (goles_local, goles_visita, ganador).

    En cancha de Mundial todo es neutral. En fase eliminatoria, si hay empate
    se resuelve con una tanda de penales (moneda ligeramente sesgada por fuerza).
    """
    mat = model.score_matrix(home, away, neutral=True)
    flat = mat.flatten()
    pick = np.random.choice(len(flat), p=flat / flat.sum())
    gh, ga = np.unravel_index(pick, mat.shape)
    gh, ga = int(gh), int(ga)

    if gh > ga:
        winner = home
    elif ga > gh:
        winner = away
    elif knockout:
        # penales: sesgo leve hacia el de mejor ataque
        bias = 0.5 + 0.04 * (model.attack[home] - model.attack[away])
        winner = home if random.random() < np.clip(bias, 0.2, 0.8) else away
    else:
        winner = None  # empate en fase de grupos
    return gh, ga, winner


def _play_group(model, teams) -> List[str]:
    """Round-robin; devuelve los equipos ordenados por puntos (desempate gol)."""
    pts = defaultdict(int)
    gd = defaultdict(int)
    gf = defaultdict(int)
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            gh, ga, winner = simulate_match(model, teams[i], teams[j])
            gd[teams[i]] += gh - ga
            gd[teams[j]] += ga - gh
            gf[teams[i]] += gh
            gf[teams[j]] += ga
            if winner == teams[i]:
                pts[teams[i]] += 3
            elif winner == teams[j]:
                pts[teams[j]] += 3
            else:
                pts[teams[i]] += 1
                pts[teams[j]] += 1
    return sorted(teams, key=lambda t: (pts[t], gd[t], gf[t]), reverse=True)


def _knockout(model, bracket: List[str]) -> str:
    """Elimina hasta que queda un campeón. `bracket` es potencia de 2."""
    teams = list(bracket)
    while len(teams) > 1:
        nxt = []
        for i in range(0, len(teams), 2):
            _, _, w = simulate_match(model, teams[i], teams[i + 1], knockout=True)
            nxt.append(w)
        teams = nxt
    return teams[0]


def simulate_tournament(model, groups: Dict[str, List[str]],
                        n_sims: int = 10000) -> Dict[str, Dict[str, float]]:
    """Corre el torneo `n_sims` veces.

    `groups` = {"A": [t1, t2, t3, t4], ...}. Avanzan los 2 primeros de cada
    grupo y se cruzan en bracket (1A-2B, 1C-2D, ...). Devuelve, por selección,
    la probabilidad de avanzar de grupo, llegar a la final y ser campeón.
    """
    champ = defaultdict(int)
    finalist = defaultdict(int)
    advance = defaultdict(int)
    group_names = list(groups.keys())

    for _ in range(n_sims):
        qualifiers = {}
        for g, teams in groups.items():
            ranked = _play_group(model, teams)
            qualifiers[g] = ranked[:2]
            for t in ranked[:2]:
                advance[t] += 1

        # bracket: 1A-2B, 1C-2D, ... (cruces estándar)
        seeds = []
        for k in range(0, len(group_names), 2):
            ga, gb = group_names[k], group_names[k + 1]
            seeds += [qualifiers[ga][0], qualifiers[gb][1],
                      qualifiers[gb][0], qualifiers[ga][1]]

        # semifinalistas para contar finalistas
        half = len(seeds) // 2
        left = _knockout(model, seeds[:half])
        right = _knockout(model, seeds[half:])
        finalist[left] += 1
        finalist[right] += 1
        _, _, w = simulate_match(model, left, right, knockout=True)
        champ[w] += 1

    all_teams = [t for ts in groups.values() for t in ts]
    return {
        t: {
            "advance": advance[t] / n_sims,
            "final": finalist[t] / n_sims,
            "champion": champ[t] / n_sims,
        }
        for t in all_teams
    }


if __name__ == "__main__":
    from data import load_results
    from dixon_coles import DixonColes

    df = load_results()
    df = df[df["date"] >= "2014-01-01"]
    model = DixonColes(half_life_days=365 * 3).fit(df)

    # Grupos de ejemplo (8 selecciones, 2 grupos) sólo para demostrar el flujo.
    groups = {
        "A": ["Brazil", "Croatia", "Mexico", "Japan"],
        "B": ["Argentina", "Spain", "Morocco", "Chile"],
    }
    res = simulate_tournament(model, groups, n_sims=2000)
    print("Selección           Avanza  Final  Campeón")
    for t, p in sorted(res.items(), key=lambda kv: kv[1]["champion"],
                       reverse=True):
        print(f"{t:<18} {p['advance']*100:5.1f}% {p['final']*100:5.1f}% "
              f"{p['champion']*100:5.1f}%")
