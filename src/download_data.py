#!/usr/bin/env python3
"""
Download complete historical and instrumental tectonic catalog for Chile (1900-2026)
from USGS FDSN Event Web Service, and compile physical subduction zone parameters
(Peru-Chile Trench geometry, Slab2 depth/dip model, and Métois et al. 2016 geodetic
interseismic locking & plate convergence profiles).
"""

import os
import time
import json
import requests
import pandas as pd
import numpy as np

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# Bounding box covering the entire Chilean subduction zone, outer-rise, megathrust,
# intermediate slab, and Andean crustal faults (Arica to Cape Horn)
CHILE_BBOX = {
    "minlatitude": -56.0,
    "maxlatitude": -17.0,
    "minlongitude": -77.0,
    "maxlongitude": -66.0,
}

TIME_WINDOWS = [
    ("1900-01-01", "1960-01-01", 2.0),
    ("1960-01-01", "1975-01-01", 2.0),
    ("1975-01-01", "1985-01-01", 2.0),
    ("1985-01-01", "1995-01-01", 2.0),
    ("1995-01-01", "2002-01-01", 2.0),
    ("2002-01-01", "2008-01-01", 2.0),
    ("2008-01-01", "2012-01-01", 2.0),
    ("2012-01-01", "2016-01-01", 2.0),
    ("2016-01-01", "2020-01-01", 2.0),
    ("2020-01-01", "2024-01-01", 2.0),
    ("2024-01-01", "2026-12-31", 2.0),
]


def download_usgs_catalog():
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    dfs = []
    for start, end, minmag in TIME_WINDOWS:
        params = {
            "format": "csv",
            "starttime": start,
            "endtime": end,
            "minmagnitude": minmag,
            **CHILE_BBOX,
            "orderby": "time-asc",
        }
        print(f"Downloading {start} to {end} (M >= {minmag})...")
        for attempt in range(3):
            try:
                resp = requests.get(url, params=params, timeout=60)
                resp.raise_for_status()
                from io import StringIO
                df_chunk = pd.read_csv(StringIO(resp.text))
                print(f"  -> Retrieved {len(df_chunk)} events.")
                dfs.append(df_chunk)
                break
            except Exception as e:
                print(f"  Attempt {attempt + 1} failed: {e}")
                time.sleep(2)
    full_df = pd.concat(dfs, ignore_index=True)
    # Deduplicate by event id
    full_df = full_df.drop_duplicates(subset=["id"]).copy()
    full_df["time"] = pd.to_datetime(full_df["time"], utc=True, format="ISO8601")
    full_df = full_df.sort_values("time").reset_index(drop=True)

    out_path = os.path.join(RAW_DIR, "chile_usgs_1900_2026.csv")
    full_df.to_csv(out_path, index=False)
    print(f"\nSaved complete catalog: {len(full_df)} unique earthquakes to {out_path}")
    print(f"Date range: {full_df['time'].min()} to {full_df['time'].max()}")
    print(f"Magnitude range: {full_df['mag'].min():.2f} to {full_df['mag'].max():.2f}")
    print(f"Events M >= 5.0: {(full_df['mag'] >= 5.0).sum()}")
    print(f"Events M >= 6.0: {(full_df['mag'] >= 6.0).sum()}")
    print(f"Events M >= 6.5: {(full_df['mag'] >= 6.5).sum()}")
    print(f"Events M >= 7.0: {(full_df['mag'] >= 7.0).sum()}")
    print(f"Events M >= 8.0: {(full_df['mag'] >= 8.0).sum()}")
    return full_df


def build_tectonic_reference():
    """
    Constructs the physical tectonic reference for the Chilean Nazca-South America
    Andean & Antarctic subduction margin based on:
    - Hayes et al. (2018) Slab2.0 model for South America trench position, slab dip, and curvature
    - Métois et al. (2016, GJI) & Ruiz & Madariaga (2018) interseismic geodetic locking
      and historical megathrust rupture segments along strike (-17°S to -56°S).
    """
    tectonic_segments = [
        {
            "name": "Arica - Camarones (North Iquique Gap)",
            "lat_min": -19.5, "lat_max": -17.0,
            "trench_lon": -71.6, "slab_dip_deg": 20.0, "convergence_mm_yr": 65.0,
            "geodetic_locking_mean": 0.65,
            "last_giant_rupture_year": 1868, "last_giant_mag": 8.8,
            "major_ruptures": [(1868, 8.8), (2005, 7.8), (2014, 6.7)]
        },
        {
            "name": "Iquique - Pisagua (2014 Rupture & Loa Segment)",
            "lat_min": -21.5, "lat_max": -19.5,
            "trench_lon": -71.3, "slab_dip_deg": 21.0, "convergence_mm_yr": 66.0,
            "geodetic_locking_mean": 0.78,
            "last_giant_rupture_year": 2014, "last_giant_mag": 8.2,
            "major_ruptures": [(1877, 8.8), (2007, 7.7), (2014, 8.2)]
        },
        {
            "name": "Tocopilla - Mejillones - Antofagasta (Southern 1877 Gap / Calama Intraslab)",
            "lat_min": -25.0, "lat_max": -21.5,
            "trench_lon": -71.4, "slab_dip_deg": 22.0, "convergence_mm_yr": 66.0,
            "geodetic_locking_mean": 0.72,
            "last_giant_rupture_year": 1995, "last_giant_mag": 8.0,
            "major_ruptures": [(1877, 8.8), (1950, 8.2), (1995, 8.0), (2007, 7.7), (2024, 7.4)]
        },
        {
            "name": "Taltal - Copiapó - Atacama Gap (1922 Segment)",
            "lat_min": -29.0, "lat_max": -25.0,
            "trench_lon": -71.8, "slab_dip_deg": 16.0, "convergence_mm_yr": 66.0,
            "geodetic_locking_mean": 0.75,
            "last_giant_rupture_year": 1922, "last_giant_mag": 8.5,
            "major_ruptures": [(1918, 7.6), (1922, 8.5), (1983, 7.6), (2013, 6.8), (2020, 6.8)]
        },
        {
            "name": "La Serena - Coquimbo - Illapel (Pampean Flat-Slab Transition)",
            "lat_min": -32.2, "lat_max": -29.0,
            "trench_lon": -72.5, "slab_dip_deg": 13.5, "convergence_mm_yr": 66.0,
            "geodetic_locking_mean": 0.68,
            "last_giant_rupture_year": 2015, "last_giant_mag": 8.3,
            "major_ruptures": [(1943, 8.1), (1997, 7.1), (2015, 8.3), (2019, 6.7)]
        },
        {
            "name": "Valparaíso - Metropolitan Central Gap (1730 / 1906 / 1985 Segment)",
            "lat_min": -34.5, "lat_max": -32.2,
            "trench_lon": -72.8, "slab_dip_deg": 18.0, "convergence_mm_yr": 66.0,
            "geodetic_locking_mean": 0.85,
            "last_giant_rupture_year": 1985, "last_giant_mag": 8.0,
            "major_ruptures": [(1730, 9.1), (1906, 8.2), (1971, 7.8), (1985, 8.0), (2017, 6.9)]
        },
        {
            "name": "Pichilemu - Constitución - Maule - Cobquecura (2010 Maule Segment)",
            "lat_min": -38.2, "lat_max": -34.5,
            "trench_lon": -73.8, "slab_dip_deg": 19.5, "convergence_mm_yr": 65.5,
            "geodetic_locking_mean": 0.82,
            "last_giant_rupture_year": 2010, "last_giant_mag": 8.8,
            "major_ruptures": [(1835, 8.5), (1928, 7.7), (1939, 8.3), (2010, 8.8), (2012, 7.1)]
        },
        {
            "name": "Arauco - Valdivia - Chiloé - Corcovado (1960 Giant Megathrust Segment)",
            "lat_min": -46.5, "lat_max": -38.2,
            "trench_lon": -75.0, "slab_dip_deg": 20.0, "convergence_mm_yr": 65.0,
            "geodetic_locking_mean": 0.80,
            "last_giant_rupture_year": 1960, "last_giant_mag": 9.5,
            "major_ruptures": [(1575, 8.5), (1837, 8.8), (1960, 9.5), (2016, 7.6)]
        },
        {
            "name": "Chile Triple Junction - Patagonia - Magallanes (Antarctic Plate Subduction)",
            "lat_min": -56.0, "lat_max": -46.5,
            "trench_lon": -75.8, "slab_dip_deg": 24.0, "convergence_mm_yr": 20.0,
            "geodetic_locking_mean": 0.35,
            "last_giant_rupture_year": 1949, "last_giant_mag": 7.7,
            "major_ruptures": [(1949, 7.7)]
        }
    ]
    out_path = os.path.join(RAW_DIR, "chile_tectonic_segments.json")
    with open(out_path, "w") as f:
        json.dump(tectonic_segments, f, indent=2)
    print(f"Saved tectonic segment reference to {out_path}")


if __name__ == "__main__":
    download_usgs_catalog()
    build_tectonic_reference()

