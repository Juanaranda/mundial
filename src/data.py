"""Descarga y carga de datos históricos de selecciones.

Fuente principal: martj42/international_results (1872 → hoy), ~47k partidos.
Es el mismo dataset que está en Kaggle, pero servido como CSV crudo desde
GitHub, así que no necesita autenticación.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    "master/results.csv"
)
SHOOTOUTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    "master/shootouts.csv"
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {url} ...")
    df = pd.read_csv(url)
    df.to_csv(dest, index=False)
    print(f"  guardado en {dest} ({len(df)} filas)")
    return dest


def load_results(refresh: bool = False) -> pd.DataFrame:
    """Devuelve el histórico de partidos con la fecha ya parseada."""
    dest = DATA_DIR / "results.csv"
    if refresh and dest.exists():
        os.remove(dest)
    _download(RESULTS_URL, dest)
    df = pd.read_csv(dest, parse_dates=["date"])
    # Normaliza tipos y descarta filas sin marcador.
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_shootouts(refresh: bool = False) -> pd.DataFrame:
    dest = DATA_DIR / "shootouts.csv"
    if refresh and dest.exists():
        os.remove(dest)
    _download(SHOOTOUTS_URL, dest)
    return pd.read_csv(dest, parse_dates=["date"])


if __name__ == "__main__":
    df = load_results(refresh=False)
    print(df.tail())
    print(f"\nTotal partidos: {len(df):,}")
    print(f"Rango: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Selecciones distintas: {df['home_team'].nunique()}")
