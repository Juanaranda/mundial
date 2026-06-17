"""Pipeline completo del pronosticador del Mundial.

  1. baja el histórico de partidos de selecciones
  2. calcula Elo dinámico (fuerza relativa)
  3. ajusta el modelo de marcadores Dixon-Coles (ponderado por recencia)
  4. aplica el ajuste por estado actual de los jugadores (si hay datos)
  5. aplica condiciones de sede (altitud, localía)
  6. predice un partido, la distribución de goles, o simula el torneo

Uso:
    python main.py partido Argentina France
    python main.py partido Bolivia Brazil --local Bolivia --sede "la paz"
    python main.py goles Argentina France
    python main.py actualizar Argentina France   # forma de jugadores vía Claude (web)
    python main.py torneo

Flags de sede (opcionales, para 'partido' y 'goles'):
    --local <equipo>   el partido NO es neutral; ese equipo juega de local
    --sede <ciudad>    resuelve la altitud por nombre (ej: "la paz", "quito")
    --alt <metros>     altitud de la sede explícita (pisa a --sede)
"""
from __future__ import annotations

import sys

from src.conditions import venue_adjustments, venue_altitude
from src.data import load_results
from src.dixon_coles import DixonColes
from src.elo import compute_elo, win_probabilities
from src.simulate import simulate_tournament
from src.squad_strength import apply_to_model, load_form_adjustments

SINCE = "2014-01-01"          # ventana de entrenamiento del modelo de goles
HALF_LIFE = 365 * 4           # vida media del peso por recencia (calibrado vs Mundial 2022)
BLEND = 0.6                   # peso Dixon-Coles vs Elo en el 1X2 (calibrado: 60% DC / 40% Elo)


def build_model():
    print("· Cargando histórico de selecciones...")
    df = load_results()

    print("· Calculando Elo dinámico...")
    df, elo = compute_elo(df)

    print("· Ajustando modelo de marcadores Dixon-Coles...")
    train = df[df["date"] >= SINCE]
    model = DixonColes(half_life_days=HALF_LIFE).fit(train)

    adj = load_form_adjustments()
    if adj:
        print(f"· Aplicando forma de jugadores ({len(adj)} selecciones)...")
        apply_to_model(model, adj)
    else:
        print("· (sin data/player_form.csv: se omite el ajuste por forma)")

    return model, elo


def _parse_venue(extra: list[str]) -> dict:
    """Lee los flags de sede de la lista de argumentos sobrantes."""
    opts = {"local": None, "city": None, "alt": None}
    i = 0
    while i < len(extra):
        if extra[i] == "--local" and i + 1 < len(extra):
            opts["local"] = extra[i + 1]; i += 2
        elif extra[i] == "--sede" and i + 1 < len(extra):
            opts["city"] = extra[i + 1]; i += 2
        elif extra[i] == "--alt" and i + 1 < len(extra):
            opts["alt"] = float(extra[i + 1]); i += 2
        else:
            i += 1
    return opts


def _resolve_conditions(home, away, opts):
    """Traduce los flags de sede a (neutral, home_log_adj, away_log_adj)."""
    neutral = opts["local"] is None
    alt = venue_altitude(city=opts["city"], altitude=opts["alt"])
    h_adj, a_adj = venue_adjustments(home, away, alt)
    return neutral, alt, h_adj, a_adj


def cmd_partido(home, away, extra):
    model, elo = build_model()
    if home not in model.attack or away not in model.attack:
        print(f"\nNo tengo suficientes datos de {home} o {away}.")
        return
    opts = _parse_venue(extra)
    neutral, alt, h_adj, a_adj = _resolve_conditions(home, away, opts)

    dc = model.predict(home, away, neutral=neutral,
                       home_log_adj=h_adj, away_log_adj=a_adj)
    ep_h, ep_d, ep_a = win_probabilities(
        elo.get(home, 1500), elo.get(away, 1500), neutral=neutral)

    sede = "neutral" if neutral else f"local: {opts['local']}"
    print(f"\n=== {home} vs {away} ({sede}, altitud {alt:.0f} m) ===")
    print(f"Elo:          {home} {elo.get(home,1500):.0f}  |  "
          f"{away} {elo.get(away,1500):.0f}")
    if alt > 0:
        print(f"Ajuste altitud (log goles): {home} {h_adj:+.3f} / "
              f"{away} {a_adj:+.3f}")
    # Modelo combinado calibrado (60% DC + 40% Elo) — el que minimiza el RPS.
    cb_h = BLEND * dc["p_home"] + (1 - BLEND) * ep_h
    cb_d = BLEND * dc["p_draw"] + (1 - BLEND) * ep_d
    cb_a = BLEND * dc["p_away"] + (1 - BLEND) * ep_a
    s = cb_h + cb_d + cb_a
    cb_h, cb_d, cb_a = cb_h / s, cb_d / s, cb_a / s

    print("                     Local   Empate  Visita")
    print(f">> COMBINADO:       {cb_h*100:5.1f}%  {cb_d*100:5.1f}%  "
          f"{cb_a*100:5.1f}%   <- predicción calibrada")
    print(f"   Dixon-Coles:     {dc['p_home']*100:5.1f}%  "
          f"{dc['p_draw']*100:5.1f}%  {dc['p_away']*100:5.1f}%")
    print(f"   Elo:             {ep_h*100:5.1f}%  {ep_d*100:5.1f}%  "
          f"{ep_a*100:5.1f}%")
    print(f"Goles esperados:    {dc['exp_home_goals']:.2f} - "
          f"{dc['exp_away_goals']:.2f}")
    print(f"Marcador más probable: {dc['most_likely_score'][0]}-"
          f"{dc['most_likely_score'][1]}")


def cmd_goles(home, away, extra):
    model, _ = build_model()
    if home not in model.attack or away not in model.attack:
        print(f"\nNo tengo suficientes datos de {home} o {away}.")
        return
    opts = _parse_venue(extra)
    neutral, alt, h_adj, a_adj = _resolve_conditions(home, away, opts)

    dist = model.goal_distribution(home, away, neutral=neutral,
                                   home_log_adj=h_adj, away_log_adj=a_adj,
                                   cap=5)
    sede = "neutral" if neutral else f"local: {opts['local']}"
    print(f"\n=== Distribución de goles · {home} vs {away} "
          f"({sede}, altitud {alt:.0f} m) ===")
    print("Goles:            0      1      2      3      4     5+")
    for team, key in [(home, "home"), (away, "away")]:
        fila = "  ".join(f"{p*100:5.1f}%" for p in dist[key])
        print(f"{team:<14} {fila}")


def cmd_actualizar(teams):
    """Actualiza data/player_form.csv buscando en la web vía Claude."""
    from src.web_update import update_player_form

    if not teams:
        print("Indicá al menos una selección: "
              "python main.py actualizar Argentina France")
        return
    update_player_form(teams)


def cmd_torneo():
    model, _ = build_model()
    # Editá estos grupos con el sorteo real del Mundial 2026.
    groups = {
        "A": ["Brazil", "Croatia", "Mexico", "Japan"],
        "B": ["Argentina", "Spain", "Morocco", "Chile"],
    }
    print("\n· Simulando torneo (10.000 corridas)...")
    res = simulate_tournament(model, groups, n_sims=10000)
    print("\nSelección           Avanza  Final  Campeón")
    for t, p in sorted(res.items(), key=lambda kv: kv[1]["champion"],
                       reverse=True):
        print(f"{t:<18} {p['advance']*100:5.1f}% {p['final']*100:5.1f}% "
              f"{p['champion']*100:5.1f}%")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    if args[0] == "partido" and len(args) >= 3:
        cmd_partido(args[1], args[2], args[3:])
    elif args[0] == "goles" and len(args) >= 3:
        cmd_goles(args[1], args[2], args[3:])
    elif args[0] == "actualizar":
        cmd_actualizar(args[1:])
    elif args[0] == "torneo":
        cmd_torneo()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
