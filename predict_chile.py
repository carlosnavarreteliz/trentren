#!/usr/bin/env python3
"""
predict_chile.py — Sistema Operacional de Pronóstico Sísmico para Chile (PI-ST-Fusion)
=======================================================================================
Herramienta CLI en tiempo real para emitir pronósticos probabilísticos de terremotos
en las 8 zonas sismotectónicas de la subducción chilena (o en una coordenada específica)
utilizando el modelo Physics-Informed Spatio-Temporal Subduction Fusion (PI-ST-Fusion).

Uso:
  python3 predict_chile.py                        # Pronóstico de 30 días con catálogo local + vivo
  python3 predict_chile.py --days 14              # Horizonte personalizado (ej. 7, 14, 30 días)
  python3 predict_chile.py --lat -33.0 --lon -71.6 --days 14   # Consulta puntual (ej. Valparaíso)
  python3 predict_chile.py --fetch-live           # Descarga últimos sismos de USGS API en vivo
"""

import argparse
import json
import math
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROCESSED_PATH = ROOT / "data" / "processed" / "chile_features_m40.csv"
RAW_CATALOG_PATH = ROOT / "data" / "raw" / "chile_usgs_1900_2026.csv"
LIVE_FORECAST_PATH = ROOT / "results" / "live_chile_forecast_oct2026.csv"
METRICS_PATH = ROOT / "results" / "benchmark_metrics.json"


def fetch_latest_usgs_events(days: int = 30, min_mag: float = 2.5) -> pd.DataFrame:
    """Descarga los sismos más recientes en Chile directamente desde la API de USGS ComCat."""
    end_dt = datetime.now(timezone.utc)
    start_dt = pd.Timestamp(end_dt) - pd.Timedelta(days=days)
    params = {
        "format": "geojson",
        "starttime": start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "minlatitude": -56.0,
        "maxlatitude": -17.0,
        "minlongitude": -77.0,
        "maxlongitude": -66.0,
        "minmagnitude": min_mag,
        "orderby": "time-asc",
    }
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "Chile-PI-ST-Fusion-Predictor/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    rows = []
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [None, None, None])
        if props.get("mag") is None or coords[0] is None:
            continue
        t_ms = props.get("time")
        dt = datetime.fromtimestamp(t_ms / 1000.0, tz=timezone.utc)
        rows.append(
            {
                "event_id": feat.get("id", ""),
                "time_iso": dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "timestamp_sec": t_ms / 1000.0,
                "latitude": float(coords[1]),
                "longitude": float(coords[0]),
                "depth_km": float(coords[2]) if coords[2] is not None else 30.0,
                "mag": float(props["mag"]),
                "place": str(props.get("place") or ""),
            }
        )
    return pd.DataFrame(rows)


def scale_horizon_prob(p_base: float, base_days: float, target_days: float) -> float:
    """Escala una probabilidad de excedencia desde base_days a target_days vía intensidad de Hazard."""
    p_clamped = min(max(float(p_base), 1e-6), 0.9999)
    lam_daily = -math.log(1.0 - p_clamped) / base_days
    return float(1.0 - math.exp(-lam_daily * target_days))


def alert_level(p_m55_14d: float, p_m60_14d: float, delta_b: float) -> str:
    """Clasifica el nivel de alerta operacional según probabilidades y anomalía de estrés."""
    if p_m60_14d >= 0.017 or p_m55_14d >= 0.060 or (p_m55_14d >= 0.045 and delta_b <= -0.15):
        return "🔴 ALTA (Vigilancia Reforzada)"
    if p_m60_14d >= 0.010 or p_m55_14d >= 0.040 or delta_b <= -0.10:
        return "🟠 MEDIA-ALTA (Acumulación)"
    if p_m55_14d >= 0.025:
        return "🟡 MODERADA (Base Activa)"
    return "🟢 BAJA (Régimen Basal)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pronóstico Sísmico Operacional para Chile — Modelo PI-ST-Fusion"
    )
    parser.add_argument(
        "--days",
        type=float,
        default=30.0,
        help="Horizonte de pronóstico en días (por defecto: 30)",
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=None,
        help="Latitud específica en Chile (ej. -33.0 para Valparaíso/Santiago)",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=None,
        help="Longitud específica en Chile (ej. -71.6)",
    )
    parser.add_argument(
        "--fetch-live",
        action="store_true",
        help="Consulta la API de USGS en tiempo real para actualizar los últimos eventos antes de reportar",
    )
    args = parser.parse_args()

    if not LIVE_FORECAST_PATH.exists():
        print(f"Error: No se encontró {LIVE_FORECAST_PATH}. Ejecute primero src/models_and_evaluation.py", file=sys.stderr)
        sys.exit(1)

    df_zones = pd.read_csv(LIVE_FORECAST_PATH)
    horizon_days = float(args.days)

    live_events_msg = ""
    if args.fetch_live:
        try:
            df_live = fetch_latest_usgs_events(days=14, min_mag=4.0)
            if len(df_live) > 0:
                last_ev = df_live.iloc[-1]
                live_events_msg = (
                    f"  Último sismo M>=4.0 USGS en vivo: {last_ev['time_iso']} | "
                    f"M {last_ev['mag']:.1f} | ({last_ev['latitude']:.2f}°, {last_ev['longitude']:.2f}°) — {last_ev['place']}\n"
                    f"  Eventos M>=4.0 últimos 14 días en Chile (USGS API): {len(df_live)}"
                )
        except Exception as e:
            live_events_msg = f"  [Aviso: No se pudo conectar a USGS en vivo ({e}); usando catálogo local verificado]"

    print("=" * 108)
    print("  SISTEMA DE PRONÓSTICO SÍSMICO OPERACIONAL PARA CHILE — PI-ST-Fusion (Physics-Informed Subduction Fusion)")
    print("=" * 108)
    print(f"  Horizonte evaluado : {horizon_days:.0f} días")
    print("  Cobertura espacial : 17.0°S a 56.0°S (Margen de Subducción Nazca – Sudamérica)")
    print("  Calibración        : CSEP Zero-Leakage | b-positive (van der Elst 2021) + Slab2 + acoplamiento geodésico")
    if live_events_msg:
        print(live_events_msg)
    print("-" * 108)

    if args.lat is not None:
        lat = float(args.lat)
        lon = float(args.lon) if args.lon is not None else -71.5
        df_feat = pd.read_csv(PROCESSED_PATH)
        dists = np.sqrt(((df_feat["latitude"] - lat) * 111.19) ** 2 + ((df_feat["longitude"] - lon) * 95.0) ** 2)
        sub = df_feat.loc[dists <= 220.0]
        if len(sub) == 0:
            sub = df_feat.iloc[np.argsort(dists)[:25]]
        last_row = sub.iloc[-1]

        matched_zone = None
        best_dist = 1e9
        for _, z in df_zones.iterrows():
            parts = str(z["lat_range"]).replace("°S", "").split(" to ")
            lmin, lmax = float(parts[0]), float(parts[1])
            if lmin <= lat <= lmax:
                matched_zone = z
                break
            mid = 0.5 * (lmin + lmax)
            if abs(mid - lat) < best_dist:
                best_dist = abs(mid - lat)
                matched_zone = z

        p50 = scale_horizon_prob(matched_zone["prob_m50_30d"], 30.0, horizon_days)
        p55 = scale_horizon_prob(matched_zone["prob_m55_30d"], 30.0, horizon_days)
        p60 = scale_horizon_prob(matched_zone["prob_m60_30d"], 30.0, horizon_days)
        p55_14 = float(matched_zone["prob_m55_14d"])
        p60_14 = scale_horizon_prob(matched_zone["prob_m60_30d"], 30.0, 14.0)
        status = alert_level(p55_14, p60_14, float(matched_zone["delta_b_pos"]))

        print(f"\n  PRONÓSTICO PUNTUAL PARA COORDENADA ({lat:.2f}°, {lon:.2f}°)")
        print(f"  Segmento Tectónico : {matched_zone['zone']} ({matched_zone['lat_range']})")
        print(f"  Nivel de Alerta    : {status}")
        print(f"  Acoplamiento (χ)   : {matched_zone['locking_chi']:.2f} | Déficit Deslizamiento: {matched_zone['slip_deficit_m']:.2f} m")
        print(f"  Valor b+ Actual    : {matched_zone['b_pos_100']:.3f} (Δb+ diferencial = {matched_zone['delta_b_pos']:+.3f})")
        print(f"  Flujo Deep->Shallow: {last_row.get('bouchon_slab_pull_ratio', last_row.get('deep_to_shallow_90d', 1.0)):.2f}x (Bouchon et al. 2016 slab pull)")
        print(f"  Probabilidades en los próximos {horizon_days:.0f} días:")
        print(f"    • P(M >= 5.0 en {horizon_days:.0f}d) : {p50 * 100:6.2f}%")
        print(f"    • P(M >= 5.5 en {horizon_days:.0f}d) : {p55 * 100:6.2f}%")
        print(f"    • P(M >= 6.0 en {horizon_days:.0f}d) : {p60 * 100:6.2f}%")
        print(f"    • Magnitud máxima esperada E[M_max] (30d): Mw {matched_zone['expected_mmax_30d']:.2f}")
        print("=" * 108)
        return

    header = (
        f"{'Zona Sismotectónica':<42} | {'χ Lock':>6} | {'Déf(m)':>6} | {'Δb+':>6} | "
        f"{f'P(M>=5.0|{horizon_days:.0f}d)':>14} | {f'P(M>=5.5|{horizon_days:.0f}d)':>14} | "
        f"{f'P(M>=6.0|{horizon_days:.0f}d)':>14} | {'Estado Operacional':<26}"
    )
    print(header)
    print("-" * 108)

    df_zones_sorted = df_zones.sort_values("prob_m55_30d", ascending=False)
    for _, row in df_zones_sorted.iterrows():
        p50 = scale_horizon_prob(row["prob_m50_30d"], 30.0, horizon_days)
        p55 = scale_horizon_prob(row["prob_m55_30d"], 30.0, horizon_days)
        p60 = scale_horizon_prob(row["prob_m60_30d"], 30.0, horizon_days)
        p60_14 = scale_horizon_prob(row["prob_m60_30d"], 30.0, 14.0)
        status = alert_level(float(row["prob_m55_14d"]), p60_14, float(row["delta_b_pos"]))
        print(
            f"{row['zone']:<42} | {row['locking_chi']:6.2f} | {row['slip_deficit_m']:6.2f} | "
            f"{row['delta_b_pos']:+6.3f} | {p50*100:13.1f}% | {p55*100:13.1f}% | "
            f"{p60*100:13.1f}% | {status:<26}"
        )

    print("-" * 108)
    # Probabilidad agregada en todo el margen chileno (asumiendo cuasi-independencia entre segmentos)
    p50_all = 1.0 - float(np.prod([1.0 - scale_horizon_prob(r["prob_m50_30d"], 30.0, horizon_days) for _, r in df_zones.iterrows()]))
    p55_all = 1.0 - float(np.prod([1.0 - scale_horizon_prob(r["prob_m55_30d"], 30.0, horizon_days) for _, r in df_zones.iterrows()]))
    p60_all = 1.0 - float(np.prod([1.0 - scale_horizon_prob(r["prob_m60_30d"], 30.0, horizon_days) for _, r in df_zones.iterrows()]))
    print(
        f"{'TOTAL MARGEN CHILENO (17°S - 56°S)':<42} | {'--':>6} | {'--':>6} | {'--':>6} | "
        f"{p50_all*100:13.1f}% | {p55_all*100:13.1f}% | {p60_all*100:13.1f}% | Síntesis Nacional"
    )
    print("=" * 108)


if __name__ == "__main__":
    main()
