"""Congela el test T1 del benchmark (solo mantenedor). Requiere forecast_short v2."""
import sys
sys.path.insert(0, ".")
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from forecast_short import load_local, build_dataset, HORIZONS_MIN, COLS_V2, MODEL_JSON_V3

OUT = Path("benchmark/t1_test.csv")


def main():
    d = load_local()
    df = build_dataset(d)
    te = df[df.time >= "2024-01-01"].reset_index(drop=True)
    model = json.loads(MODEL_JSON_V3.read_text())
    mean = np.array(model["scaler_mean"])
    scale = np.array(model["scaler_scale"])
    Xs = (te[COLS_V2].to_numpy() - mean) / scale
    out = pd.DataFrame({"trigger_idx": te.index, "time": te.time, "lat": te.lat,
                        "lon": te.lon, "mag": te.x2})
    tr = df[df.time < "2021-01-01"]
    for h in HORIZONS_MIN:
        m = model["horizons"][str(h)]
        s = Xs @ np.array(m["coef"]) + m["intercept"]
        p = 1 / (1 + np.exp(-(m["platt_coef"] * s + m["platt_intercept"])))
        out[f"y_{h}"] = te[f"y_{h}"].to_numpy()
        out[f"p_sota_{h}"] = p
        out[f"p_poisson_{h}"] = float(tr[f"y_{h}"].mean())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"test congelado: {len(out)} triggers -> {OUT}")
    for h in HORIZONS_MIN:
        print(f"  H={h}: pos={int(out[f'y_{h}'].sum())} sota_mean={out[f'p_sota_{h}'].mean():.4f}")


if __name__ == "__main__":
    sys.exit(main())
