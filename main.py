"""Pipeline completo del pronosticador del Mundial.

  1. baja el histórico de partidos de selecciones
  2. calcula Elo dinámico (fuerza relativa)
  3. ajusta el modelo de marcadores Dixon-Coles (ponderado por recencia)
  4. aplica el ajuste por estado actual de los jugadores (si hay datos)
  5. predice un partido y/o simula el torneo

Uso:
    python main.py partido Argentina Francia
    python main.py torneo
"""
from __future__ import annotations

import sys

from src.data import load_results
from src.dixon_coles import DixonColes
from src.elo import compute_elo, win_probabilities
from src.simulate import simulate_tournament
from src.squad_strength import apply_to_model, load_form_adjustments

SINCE = "2014-01-01"          # ventana de entrenamiento del modelo de goles
HALF_LIFE = 365 * 3           # vida media del peso por recencia (días)


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


def cmd_partido(home: str, away: str):
    model, elo = build_model()
    if home not in model.attack or away not in model.attack:
        print(f"\nNo tengo suficientes datos de {home} o {away}.")
        return
    dc = model.predict(home, away, neutral=True)
    eh, _, ea = elo.get(home, 1500), 0, elo.get(away, 1500)
    ep_h, ep_d, ep_a = win_probabilities(eh, ea, neutral=True)

    print(f"\n=== {home} vs {away} (cancha neutral) ===")
    print(f"Elo:          {home} {elo.get(home,1500):.0f}  |  "
          f"{away} {elo.get(away,1500):.0f}")
    print("                     Local   Empate  Visita")
    print(f"Dixon-Coles:        {dc['p_home']*100:5.1f}%  "
          f"{dc['p_draw']*100:5.1f}%  {dc['p_away']*100:5.1f}%")
    print(f"Elo:                {ep_h*100:5.1f}%  {ep_d*100:5.1f}%  "
          f"{ep_a*100:5.1f}%")
    print(f"Goles esperados:    {dc['exp_home_goals']:.2f} - "
          f"{dc['exp_away_goals']:.2f}")
    print(f"Marcador más probable: {dc['most_likely_score'][0]}-"
          f"{dc['most_likely_score'][1]}")


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
        cmd_partido(args[1], args[2])
    elif args[0] == "torneo":
        cmd_torneo()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
