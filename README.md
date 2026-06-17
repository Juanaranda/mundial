# mundial — pronosticador del Mundial 🏆

Predictor de partidos y simulador del Mundial que combina **dos señales**:

1. **Fuerza histórica de las selecciones** — historial de resultados +
   enfrentamientos directos, vía un modelo de marcadores **Dixon-Coles**
   (Poisson bivariado con corrección para marcadores bajos) ponderado por
   recencia, más un **Elo dinámico**.
2. **Estado actual de los jugadores** — la forma reciente de los convocados
   (rating en el club, lesiones, disponibilidad) ajusta el ataque/defensa de
   cada selección.
3. **Condiciones de la sede** — altitud y localía ajustan los goles esperados
   por encima de la fuerza histórica (Bolivia en La Paz es el caso canónico).

Con eso se simula el torneo miles de veces (Monte Carlo) para estimar la
probabilidad de que cada selección avance de fase y sea campeón.

## De dónde sale la data

| Qué | Fuente | Notas |
|-----|--------|-------|
| Resultados históricos (1872→hoy, ~47k partidos) | [martj42/international_results](https://github.com/martj42/international_results) (mirror del dataset de Kaggle) | se baja solo, sin auth |
| Elo histórico | se calcula en `src/elo.py` desde el histórico | — |
| Fixtures / alineaciones / lesiones | [API-Football](https://www.api-sports.io/) o [Football-Data.org](https://www.football-data.org/) | requiere API key |
| Rating por jugador y partido | SofaScore / FotMob | no oficial, scraping |
| Ranking de referencia | [FIFA](https://www.fifa.com/fifa-world-ranking) / [eloratings.net](https://eloratings.net) | opcional |

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
# Predecir un partido (cancha neutral)
python main.py partido Argentina France

# Partido con sede: Bolivia de local en La Paz (3640 m)
python main.py partido Bolivia Brazil --local Bolivia --sede "la paz"

# Distribución de goles: P(cada equipo marque 0,1,2,3,4,5+)
python main.py goles Argentina France

# Simular el torneo completo
python main.py torneo
```

### Flags de sede (para `partido` y `goles`)

| Flag | Qué hace |
|------|----------|
| `--local <equipo>` | el partido NO es neutral; ese equipo juega de local |
| `--sede <ciudad>`  | resuelve la altitud por nombre (`"la paz"`, `"quito"`, `"mexico city"`...) |
| `--alt <metros>`   | altitud de la sede explícita (pisa a `--sede`) |

## Estructura

```
mundial/
├── main.py                 # pipeline: data → Elo → Dixon-Coles → forma → simulación
├── src/
│   ├── data.py             # descarga/carga del histórico
│   ├── elo.py              # Elo dinámico por selección
│   ├── dixon_coles.py      # modelo de marcadores + distribución de goles
│   ├── squad_strength.py   # ajuste por forma de los jugadores
│   ├── conditions.py       # condiciones de sede (altitud, localía)
│   └── simulate.py         # Monte Carlo del torneo
└── data/
    └── player_form.csv     # (lo poblás vos) forma de los jugadores
```

## Estado de los jugadores (forma)

Copiá `data/player_form.csv.example` a `data/player_form.csv` y llenalo con la
forma de los convocados:

```csv
team,player,position,recent_rating,baseline_rating,available
Argentina,Lionel Messi,FWD,8.4,7.9,1
```

- `recent_rating`: rating promedio en sus últimos partidos de club (0–10).
- `baseline_rating`: su nivel base / temporada.
- `available`: 1 si está disponible, 0 si lesionado/suspendido.

El módulo convierte la diferencia `recent − baseline` en un ajuste sobre el
ataque (DEL/MED) y la defensa (DEF/GK) de cada selección. Si el archivo no
existe, el modelo corre igual sin ese ajuste.

## Próximos pasos sugeridos

- **Backtesting**: entrenar con datos hasta 2018 y validar contra el Mundial
  2022 para calibrar `half_life`, `BETA_ATK/BETA_DEF` y la ventaja de localía.
- Conectar API-Football para poblar `player_form.csv` automáticamente.
- Reemplazar/blender Dixon-Coles con un **XGBoost** que use Elo, forma,
  descanso e importancia del partido como features.
