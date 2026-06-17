"""Condiciones de la sede: altitud y localía.

Estas variables ajustan los goles esperados *encima* de la fuerza histórica
que ya estima Dixon-Coles. La idea es devolver ajustes en escala logarítmica
(se suman al log de la tasa de goles de cada equipo), para que se acoplen
limpio al modelo Poisson.

  * ALTITUD: un equipo acostumbrado al nivel del mar que juega en altura rinde
    menos (menos oxígeno, la pelota viaja distinto). Penalizamos sólo cuando un
    equipo SUBE respecto a su altura habitual. Bolivia en La Paz es el caso
    canónico.
  * LOCALÍA: si el partido NO es en cancha neutral, el local recibe la ventaja
    `home_adv` que ya estima el modelo (se maneja con el flag `neutral`).

Los coeficientes son interpretables pero hay que CALIBRARLOS con backtesting.
"""
from __future__ import annotations

# Altitud (m sobre el nivel del mar) de sedes relevantes / históricas.
CITY_ALTITUDE = {
    "la paz": 3640, "quito": 2850, "bogota": 2640, "bogotá": 2640,
    "mexico city": 2240, "ciudad de mexico": 2240, "ciudad de méxico": 2240,
    "toluca": 2660, "cusco": 3400, "addis ababa": 2355, "sucre": 2810,
    "johannesburg": 1753, "pretoria": 1339, "denver": 1609,
    "guadalajara": 1566, "san jose": 1172, "asuncion": 43,
    # sedes Mundial 2026 (todas relativamente bajas, referencia)
    "mexico city 2026": 2240, "guadalajara 2026": 1566, "monterrey": 540,
    "denver 2026": 1609, "atlanta": 320, "kansas city": 277,
}

# Altura "de casa" aproximada de cada selección (m). Default: nivel del mar.
TEAM_HOME_ALTITUDE = {
    "Bolivia": 3640, "Ecuador": 2850, "Colombia": 2640, "Mexico": 2240,
    "Peru": 150, "Iran": 1200, "Afghanistan": 1790, "Ethiopia": 2355,
    "Saudi Arabia": 600, "South Africa": 1450, "Switzerland": 500,
    "Austria": 200, "Bhutan": 2330,
}
DEFAULT_HOME_ALTITUDE = 50.0

# Cuánto castiga subir 1000 m por encima de lo habitual (en log de goles).
# 0.06 ≈ -6% de goles por cada 1000 m de desnivel. CALIBRAR con datos.
ALT_K = 0.06


def venue_altitude(city: str | None = None, altitude: float | None = None) -> float:
    """Resuelve la altitud de la sede por nombre de ciudad o valor explícito."""
    if altitude is not None:
        return float(altitude)
    if city:
        return float(CITY_ALTITUDE.get(city.strip().lower(), 0.0))
    return 0.0


def _altitude_penalty(team: str, venue_alt: float) -> float:
    """Ajuste log (<=0) a la tasa de goles del equipo por jugar en altura."""
    home_alt = TEAM_HOME_ALTITUDE.get(team, DEFAULT_HOME_ALTITUDE)
    climb = max(0.0, venue_alt - home_alt)  # sólo penaliza subir
    return -ALT_K * (climb / 1000.0)


def venue_adjustments(home: str, away: str, venue_alt: float = 0.0
                      ) -> tuple[float, float]:
    """Devuelve (ajuste_log_local, ajuste_log_visita) por altitud.

    Se suman al log de lambda (local) y mu (visita) en el modelo de goles.
    """
    return _altitude_penalty(home, venue_alt), _altitude_penalty(away, venue_alt)


if __name__ == "__main__":
    for h, a, city in [("Bolivia", "Brazil", "la paz"),
                       ("Brazil", "Bolivia", "rio"),
                       ("Argentina", "France", "monterrey")]:
        alt = venue_altitude(city=city)
        ah, aa = venue_adjustments(h, a, alt)
        print(f"{h} vs {a} en {city} ({alt:.0f} m): "
              f"local {ah:+.3f} / visita {aa:+.3f} (log goles)")
