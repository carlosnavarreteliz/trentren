"""Parse raw MUSTANG noise-mode files into one long parquet: date, target, freq, power_db."""
import sys, io
from pathlib import Path
import pandas as pd

def parse_file(fn):
    txt = Path(fn).read_text()
    if not txt.startswith("target:"):
        return None
    lines = txt.splitlines()
    tgt = lines[0].split(":", 1)[1].strip()
    i = [k for k, l in enumerate(lines) if l.startswith("# frequencies")][0]
    freqs = [float(x) for x in lines[i + 1].split(",")]
    rows = [l for l in lines[i + 3:] if l[:2] in ("19", "20")]
    df = pd.read_csv(io.StringIO("\n".join(rows)), header=None)
    df.columns = ["date"] + [f"{f:.4g}" for f in freqs]
    df["date"] = pd.to_datetime(df["date"])
    df = df.melt(id_vars="date", var_name="freq", value_name="db")
    df["freq"] = df["freq"].astype(float)
    df["target"] = tgt
    return df

if __name__ == "__main__":
    raw = Path(sys.argv[1]); outp = Path(sys.argv[2])
    dfs = [d for d in (parse_file(f) for f in sorted(raw.glob("*.txt"))) if d is not None]
    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet(outp)
    print(df.shape, df.target.nunique(), sorted(df.freq.unique()))
