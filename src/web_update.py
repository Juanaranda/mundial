"""Actualización vía web usando Claude (sin APIs deportivas externas).

En vez de depender de API-Football / SofaScore, le pedimos a **Claude** que
busque en la web la forma reciente, lesiones y disponibilidad de los jugadores
convocados, y escriba `data/player_form.csv` listo para el modelo.

Usa la herramienta de búsqueda web del lado del servidor de la Messages API
(`web_search_20260209`): Claude ejecuta las búsquedas en la infraestructura de
Anthropic y devuelve el resultado ya procesado. La única dependencia es una
clave de API de Claude.

Requisitos:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...     # NO la pegues en el código

Uso:
    python -m src.web_update Argentina France Brazil
    # o desde main.py:  python main.py actualizar Argentina France
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL = "claude-opus-4-8"

PROMPT = """Buscá en la web el estado ACTUAL de los jugadores clave convocables \
de la selección de fútbol de {team} (datos de las últimas semanas: rendimiento \
en su club, lesiones, suspensiones, minutos jugados).

Devolvé ÚNICAMENTE un array JSON (sin texto alrededor, sin markdown) con los \
~15 jugadores más relevantes, con este formato exacto por jugador:
  {{"player": "Nombre", "position": "FWD|MID|DEF|GK", \
"recent_rating": <0-10>, "baseline_rating": <0-10>, "available": 0 o 1}}

- recent_rating: nivel mostrado en sus últimos partidos (0-10).
- baseline_rating: su nivel habitual / de temporada (0-10).
- available: 1 si está disponible para jugar, 0 si está lesionado o suspendido.
Usá las posiciones en inglés abreviadas: FWD, MID, DEF, GK."""

WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search"}


def _extract_json_array(text: str) -> list[dict]:
    """Saca el primer array JSON del texto (tolera ruido alrededor)."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return []


def fetch_team_form(team: str, model: str = MODEL) -> list[dict]:
    """Pide a Claude (con búsqueda web) la forma de los jugadores de `team`."""
    import anthropic  # import perezoso: el resto del proyecto no lo necesita

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "Falta ANTHROPIC_API_KEY. Exportala: export ANTHROPIC_API_KEY=sk-ant-..."
        )

    client = anthropic.Anthropic()
    # Streaming para no chocar con timeouts mientras corre la búsqueda web.
    with client.messages.stream(
        model=model,
        max_tokens=8000,
        thinking={"type": "adaptive"},
        tools=[WEB_SEARCH_TOOL],
        messages=[{"role": "user", "content": PROMPT.format(team=team)}],
    ) as stream:
        msg = stream.get_final_message()

    if msg.stop_reason == "refusal":
        print(f"  [{team}] Claude declinó la consulta; se omite.")
        return []

    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    players = _extract_json_array(text)
    for p in players:
        p["team"] = team
    return players


def update_player_form(teams: list[str],
                       out_path: str | Path | None = None) -> Path:
    """Genera/actualiza data/player_form.csv para las selecciones dadas."""
    import csv

    out_path = Path(out_path) if out_path else DATA_DIR / "player_form.csv"
    fields = ["team", "player", "position", "recent_rating",
              "baseline_rating", "available"]

    rows: list[dict] = []
    for team in teams:
        print(f"· Buscando forma de jugadores de {team} (vía Claude)...")
        rows.extend(fetch_team_form(team))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ {len(rows)} jugadores escritos en {out_path}")
    return out_path


if __name__ == "__main__":
    import sys

    teams = sys.argv[1:] or ["Argentina", "France"]
    update_player_form(teams)
