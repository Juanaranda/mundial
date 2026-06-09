"""Modelo Dixon-Coles para predecir marcadores entre selecciones.

Es el clásico de la literatura de apuestas deportivas: un Poisson bivariado
con corrección para marcadores bajos (0-0, 1-0, 0-1, 1-1) y ponderación por
recencia (los partidos viejos pesan menos).

Para cada partido local i vs visita j:
    goles_local  ~ Poisson( exp(ataque_i - defensa_j + ventaja_local) )
    goles_visita ~ Poisson( exp(ataque_j - defensa_i) )

Se ajusta maximizando la verosimilitud con `scipy.optimize`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


def _tau(x, y, lam, mu, rho):
    """Corrección Dixon-Coles para marcadores bajos."""
    out = np.ones_like(lam, dtype=float)
    out = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, out)
    out = np.where((x == 0) & (y == 1), 1 + lam * rho, out)
    out = np.where((x == 1) & (y == 0), 1 + mu * rho, out)
    out = np.where((x == 1) & (y == 1), 1 - rho, out)
    return out


@dataclass
class DixonColes:
    half_life_days: float = 365 * 2  # peso a la mitad tras 2 años
    teams: List[str] = field(default_factory=list)
    attack: Dict[str, float] = field(default_factory=dict)
    defense: Dict[str, float] = field(default_factory=dict)
    home_adv: float = 0.0
    rho: float = 0.0

    def _weights(self, dates: pd.Series, ref_date: pd.Timestamp) -> np.ndarray:
        age_days = (ref_date - dates).dt.days.to_numpy(dtype=float)
        return 0.5 ** (age_days / self.half_life_days)

    def fit(self, df: pd.DataFrame, ref_date: pd.Timestamp | None = None,
            min_matches: int = 8) -> "DixonColes":
        ref_date = ref_date or df["date"].max()

        # Sólo selecciones con suficiente muestra reciente.
        counts = pd.concat([df["home_team"], df["away_team"]]).value_counts()
        keep = set(counts[counts >= min_matches].index)
        df = df[df["home_team"].isin(keep) & df["away_team"].isin(keep)].copy()

        self.teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)

        hi = df["home_team"].map(idx).to_numpy()
        ai = df["away_team"].map(idx).to_numpy()
        hs = df["home_score"].to_numpy()
        as_ = df["away_score"].to_numpy()
        w = self._weights(df["date"], ref_date)

        # Parámetros: [ataque(n), defensa(n), home_adv, rho]
        # Se fija la media de ataque en 0 para identificabilidad.
        init = np.concatenate([
            np.zeros(n), np.zeros(n), np.array([0.25, -0.05])
        ])

        def neg_log_lik(params):
            atk = params[:n]
            dfn = params[n:2 * n]
            home_adv = params[2 * n]
            rho = params[2 * n + 1]

            lam = np.exp(atk[hi] - dfn[ai] + home_adv)
            mu = np.exp(atk[ai] - dfn[hi])
            lam = np.clip(lam, 1e-6, 12)
            mu = np.clip(mu, 1e-6, 12)

            ll = (poisson.logpmf(hs, lam) + poisson.logpmf(as_, mu)
                  + np.log(np.clip(_tau(hs, as_, lam, mu, rho), 1e-9, None)))
            # Penalización suave para fijar media de ataque ~ 0.
            penalty = 1000 * (atk.mean() ** 2)
            return -np.sum(w * ll) + penalty

        res = minimize(neg_log_lik, init, method="L-BFGS-B",
                       options={"maxiter": 400})

        p = res.x
        self.attack = {t: p[i] for t, i in idx.items()}
        self.defense = {t: p[n + i] for t, i in idx.items()}
        self.home_adv = float(p[2 * n])
        self.rho = float(p[2 * n + 1])
        return self

    # ---- predicción ----
    def score_matrix(self, home: str, away: str, neutral: bool = True,
                     max_goals: int = 10) -> np.ndarray:
        """Matriz de probabilidad P(goles_local=x, goles_visita=y)."""
        adv = 0.0 if neutral else self.home_adv
        lam = np.exp(self.attack[home] - self.defense[away] + adv)
        mu = np.exp(self.attack[away] - self.defense[home])
        lam, mu = min(lam, 12), min(mu, 12)

        x = np.arange(0, max_goals + 1)
        ph = poisson.pmf(x, lam)
        pa = poisson.pmf(x, mu)
        mat = np.outer(ph, pa)

        # corrección DC en las 4 celdas bajas
        for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            mat[i, j] *= _tau(np.array(i), np.array(j), lam, mu, self.rho)
        return mat / mat.sum()

    def predict(self, home: str, away: str, neutral: bool = True) -> dict:
        mat = self.score_matrix(home, away, neutral)
        p_home = np.tril(mat, -1).sum()
        p_draw = np.trace(mat)
        p_away = np.triu(mat, 1).sum()
        # marcador más probable
        i, j = np.unravel_index(mat.argmax(), mat.shape)
        return {
            "home": home, "away": away,
            "p_home": float(p_home), "p_draw": float(p_draw),
            "p_away": float(p_away),
            "exp_home_goals": float((mat.sum(1) * np.arange(mat.shape[0])).sum()),
            "exp_away_goals": float((mat.sum(0) * np.arange(mat.shape[1])).sum()),
            "most_likely_score": (int(i), int(j)),
        }


if __name__ == "__main__":
    from data import load_results

    df = load_results()
    df = df[df["date"] >= "2014-01-01"]  # ventana reciente => ajuste rápido
    model = DixonColes(half_life_days=365 * 3).fit(df)
    print(f"Ventaja de localía estimada: {model.home_adv:.3f}")
    print(f"rho (corrección DC): {model.rho:.3f}\n")
    for h, a in [("Brazil", "Argentina"), ("Spain", "France"),
                 ("Chile", "Argentina"), ("Morocco", "Portugal")]:
        if h in model.attack and a in model.attack:
            r = model.predict(h, a, neutral=True)
            print(f"{h} vs {a}: "
                  f"L {r['p_home']*100:4.1f}% / E {r['p_draw']*100:4.1f}% / "
                  f"V {r['p_away']*100:4.1f}%  | marcador prob. "
                  f"{r['most_likely_score'][0]}-{r['most_likely_score'][1]}")
