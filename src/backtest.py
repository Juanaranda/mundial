"""Backtesting y calibración del pronosticador.

Mide qué tan bien predice el modelo sobre partidos que NO vio al entrenar
(out-of-sample), usando métricas estándar de la literatura de pronóstico
deportivo, y calibra los hiperparámetros para minimizar el error.

Métricas:
  * RPS (Ranked Probability Score): la métrica de referencia para 1-X-2.
    Penaliza más si te equivocás "lejos" (predecir local cuando ganó visita
    es peor que predecir empate). Más bajo = mejor. Casas de apuestas ≈ 0.19.
  * log-loss: castiga fuerte la sobreconfianza equivocada.
  * accuracy: % de veces que el resultado más probable fue el que pasó.
  * calibración: cuando decís 60%, ¿pasa ~60% de las veces?

Regla anti-leakage: el modelo Dixon-Coles se entrena SOLO con partidos
anteriores a la fecha de test. El Elo usa el rating PREVIO a cada partido
(columnas home_elo/away_elo), que ya es leak-free por construcción.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dixon_coles import DixonColes
from .elo import compute_elo, win_probabilities


def result_char(hs: int, as_: int) -> str:
    if hs > as_:
        return "H"
    if hs < as_:
        return "A"
    return "D"


def rps(p_home: float, p_draw: float, p_away: float, outcome: str) -> float:
    """Ranked Probability Score para resultado ordenado [H, D, A]."""
    p = [p_home, p_draw, p_away]
    e = {"H": [1, 0, 0], "D": [0, 1, 0], "A": [0, 0, 1]}[outcome]
    cum_p = cum_e = 0.0
    total = 0.0
    for i in range(2):  # r-1 = 2 términos
        cum_p += p[i]
        cum_e += e[i]
        total += (cum_p - cum_e) ** 2
    return total / 2.0


def _metrics(preds: list[dict]) -> dict:
    rps_v, ll_v, hits = [], [], 0
    for r in preds:
        rps_v.append(rps(r["p_home"], r["p_draw"], r["p_away"], r["outcome"]))
        probs = {"H": r["p_home"], "D": r["p_draw"], "A": r["p_away"]}
        ll_v.append(-np.log(max(probs[r["outcome"]], 1e-12)))
        if max(probs, key=probs.get) == r["outcome"]:
            hits += 1
    n = len(preds)
    return {"n": n, "rps": float(np.mean(rps_v)),
            "logloss": float(np.mean(ll_v)), "accuracy": hits / n if n else 0.0}


def _calibration_table(preds: list[dict], bins=(0, .2, .4, .6, .8, 1.01)) -> str:
    """Agrupa todas las probabilidades emitidas y compara con la frecuencia real."""
    rows = []
    obs = []  # (prob_predicha, ocurrió 0/1)
    for r in preds:
        for k, out in [("p_home", "H"), ("p_draw", "D"), ("p_away", "A")]:
            obs.append((r[k], 1 if r["outcome"] == out else 0))
    obs = np.array(obs)
    out = ["  prob predicha | real | n"]
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (obs[:, 0] >= lo) & (obs[:, 0] < hi)
        if mask.sum() == 0:
            continue
        pred_mean = obs[mask, 0].mean()
        real = obs[mask, 1].mean()
        out.append(f"  {lo:.0%}-{hi:.0%}: {pred_mean:5.1%} | {real:5.1%} | "
                   f"{int(mask.sum())}")
    return "\n".join(out)


def predict_set(df_all: pd.DataFrame, test: pd.DataFrame, cutoff: str,
                half_life_days: float, since: str = "2014-01-01",
                blend: float = 1.0) -> list[dict]:
    """Entrena DC con datos < cutoff y predice los partidos de `test`.

    `blend`: peso de Dixon-Coles vs Elo en la probabilidad final (1.0 = solo DC,
    0.0 = solo Elo).
    """
    train = df_all[(df_all["date"] >= since) & (df_all["date"] < cutoff)]
    model = DixonColes(half_life_days=half_life_days).fit(train)

    preds = []
    for row in test.itertuples(index=False):
        h, a = row.home_team, row.away_team
        if h not in model.attack or a not in model.attack:
            continue
        neutral = bool(getattr(row, "neutral", True))
        dc = model.predict(h, a, neutral=neutral)
        eh, ea = row.home_elo, row.away_elo
        e_h, e_d, e_a = win_probabilities(eh, ea, neutral=neutral)

        p_h = blend * dc["p_home"] + (1 - blend) * e_h
        p_d = blend * dc["p_draw"] + (1 - blend) * e_d
        p_a = blend * dc["p_away"] + (1 - blend) * e_a
        s = p_h + p_d + p_a
        preds.append({
            "p_home": p_h / s, "p_draw": p_d / s, "p_away": p_a / s,
            "outcome": result_char(row.home_score, row.away_score),
        })
    return preds


def baseline_metrics(test: pd.DataFrame) -> dict:
    """Baseline tonto: usa las frecuencias base de local/empate/visita."""
    preds = [{"p_home": .455, "p_draw": .265, "p_away": .28,
              "outcome": result_char(r.home_score, r.away_score)}
             for r in test.itertuples(index=False)]
    return _metrics(preds)


# ---------------------------------------------------------------------------
# Runners de calibración
# ---------------------------------------------------------------------------
HALF_LIVES = [365 * 1.5, 365 * 2, 365 * 3, 365 * 4]
BLENDS = [1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.0]


def _print_metrics(name: str, m: dict):
    print(f"  {name:<24} RPS {m['rps']:.4f} | logloss {m['logloss']:.3f} | "
          f"acc {m['accuracy']:.1%} | n={m['n']}")


def calibrate_wc2022(df: pd.DataFrame):
    """Calibra contra el Mundial 2022 (out-of-sample puro)."""
    wc = df[(df["tournament"] == "FIFA World Cup") &
            (df["date"] >= "2022-11-01") & (df["date"] <= "2022-12-31")]
    cutoff = "2022-11-19"
    print(f"\n=== Mundial 2022 ({len(wc)} partidos) — calibrando half_life ===")
    print(f"Baseline (frecuencias base):")
    _print_metrics("baseline", baseline_metrics(wc))

    best_hl, best_rps, best_preds = None, 1e9, None
    for hl in HALF_LIVES:
        preds = predict_set(df, wc, cutoff, hl, blend=1.0)
        m = _metrics(preds)
        _print_metrics(f"DC half_life={hl/365:.1f}a", m)
        if m["rps"] < best_rps:
            best_hl, best_rps, best_preds = hl, m["rps"], preds

    print(f"\n  → mejor half_life: {best_hl/365:.1f} años (RPS {best_rps:.4f})")

    print(f"\n=== Mezcla Dixon-Coles ↔ Elo (half_life={best_hl/365:.1f}a) ===")
    best_blend, best_brps = 1.0, 1e9
    for w in BLENDS:
        preds = predict_set(df, wc, cutoff, best_hl, blend=w)
        m = _metrics(preds)
        tag = "solo DC" if w == 1 else ("solo Elo" if w == 0 else f"{w:.0%} DC")
        _print_metrics(tag, m)
        if m["rps"] < best_brps:
            best_blend, best_brps = w, m["rps"]
    print(f"\n  → mejor mezcla: {best_blend:.0%} DC (RPS {best_brps:.4f})")

    final = predict_set(df, wc, cutoff, best_hl, blend=best_blend)
    print(f"\n=== Calibración de la mejor config ===")
    print(_calibration_table(final))
    return best_hl, best_blend


def calibrate_recent(df: pd.DataFrame, half_life: float, blend: float):
    """Valida sobre clasificatorias y torneos recientes (2023→2026),
    con folds anuales (entrena con datos previos a cada año)."""
    competitive = df[~df["tournament"].isin(["Friendly"])]
    print(f"\n=== Validación 2023-2026 (clasificatorias + torneos) ===")
    all_preds = []
    for year in [2023, 2024, 2025, 2026]:
        test = competitive[(competitive["date"] >= f"{year}-01-01") &
                           (competitive["date"] < f"{year+1}-01-01")]
        if test.empty:
            continue
        preds = predict_set(df, test, f"{year}-01-01", half_life, blend=blend)
        m = _metrics(preds)
        _print_metrics(f"{year} ({len(preds)} part.)", m)
        all_preds.extend(preds)
    print("  " + "-" * 50)
    _print_metrics("TOTAL", _metrics(all_preds))
    _print_metrics("baseline TOTAL", baseline_metrics(competitive[
        competitive["date"] >= "2023-01-01"]))
    print("\n=== Calibración global ===")
    print(_calibration_table(all_preds))


if __name__ == "__main__":
    import sys
    from src.data import load_results

    print("· Cargando histórico y calculando Elo...")
    df = load_results()
    df, _ = compute_elo(df)

    mode = sys.argv[1] if len(sys.argv) > 1 else "wc2022"
    if mode == "wc2022":
        calibrate_wc2022(df)
    elif mode == "recent":
        hl = float(sys.argv[2]) if len(sys.argv) > 2 else 365 * 3
        bl = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
        calibrate_recent(df, hl, bl)
