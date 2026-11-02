"""Run PhaseNetLight (SeisBench, pretrained 'stead') over continuous 20 Hz IPOC data, day by day.

Usage: /opt/miniconda3/bin/python3 -I pick.py <mseed_root> <out_dir> <window> <station> [P_thr] [S_thr]

Per station-day:  bandpass 1-9 Hz at 20 Hz -> resample_poly x5 to 100 Hz -> 3001-sample windows,
stride 2500 (blinding 250 each side gives contiguous coverage) -> per-component demean/peak norm (as SeisBench annotate_batch_pre for these v2 weights) ->
PhaseNetLight forward -> stitched P/S probability traces -> peaks above threshold (min 1 s apart).
Windows with >5 % zero/flat samples in any component are skipped (gap); uptime is recorded.
Output: <out_dir>/picks_<window>_<station>.parquet and uptime_<window>_<station>.csv
The last day of each window is truncated at 23:46:00 UTC (Iquique M8.2 origin 23:46:47 on 2014-04-01).
"""
import sys, os, glob, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, sosfiltfilt, resample_poly, find_peaks

torch.set_num_threads(int(os.environ.get("PICK_THREADS", "1")))
import obspy
import seisbench.models as sbm

root, outd, win, sta = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
P_THR = float(sys.argv[5]) if len(sys.argv) > 5 else 0.3
S_THR = float(sys.argv[6]) if len(sys.argv) > 6 else 0.3
outd.mkdir(parents=True, exist_ok=True)

model = sbm.PhaseNetLight.from_pretrained("stead", update=True).eval()
assert model.norm == "peak" and not model.norm_detrend
LAB = model.labels  # 'NPS'
iP, iS = LAB.index("P"), LAB.index("S")
FS, UP = 20.0, 5
NW, STRIDE, BL = 3001, 2500, 250
SOS = butter(4, [1.0, 9.0], btype="band", fs=FS, output="sos")


def day_array(fn, day0):
    """Return (3, N) float32 array on the day grid (ZNE order), and a validity mask."""
    n = int(86400 * FS)
    arr = np.zeros((3, n), np.float32); ok = np.zeros((3, n), bool)
    try:
        st = obspy.read(str(fn))
    except Exception:
        return arr, ok
    for tr in st:
        if tr.stats.sampling_rate != FS:
            if tr.stats.sampling_rate == 40.0:
                tr.decimate(2, no_filter=False)
            else:
                continue
    for ci, c in enumerate("ZNE"):
        sub = st.select(component=c)
        for tr in sub:
            if tr.stats.sampling_rate != FS:
                continue
            i0 = int(round((tr.stats.starttime - day0) * FS))
            d = tr.data.astype(np.float64)
            if len(d) < 200:
                continue
            d = d - np.mean(d)
            d = sosfiltfilt(SOS, d).astype(np.float32)
            a, b = max(i0, 0), min(i0 + len(d), n)
            if b <= a:
                continue
            arr[ci, a:b] = d[a - i0:b - i0]
            ok[ci, a:b] = True
    return arr, ok


def run_day(fn, day0, tcut=None):
    arr, ok = day_array(fn, day0)
    nday = arr.shape[1]
    if tcut is not None:
        ncut = int((tcut - day0) * FS)
        ok[:, ncut:] = False; arr[:, ncut:] = 0
    if ok.sum() == 0:
        return [], 0.0
    x = resample_poly(arr, UP, 1, axis=1).astype(np.float32)  # (3, N*5)
    okx = np.repeat(ok, UP, axis=1)
    N = x.shape[1]
    starts = np.arange(0, N - NW + 1, STRIDE)
    if starts[-1] != N - NW:
        starts = np.append(starts, N - NW)
    probP = np.zeros(N, np.float32); probS = np.zeros(N, np.float32)
    valid_samples = 0
    good = [s for s in starts if okx[:, s:s + NW].mean(axis=1).min() > 0.95]
    for b0 in range(0, len(good), 512):
        idx = good[b0:b0 + 512]
        batch = np.stack([x[:, s:s + NW] for s in idx])
        batch = batch - batch.mean(axis=-1, keepdims=True)
        batch = batch / (np.abs(batch).max(axis=-1, keepdims=True) + 1e-10)
        with torch.no_grad():
            y = model(torch.from_numpy(batch)).numpy()
        for k, s in enumerate(idx):
            a, b = s + BL, s + NW - BL
            probP[a:b] = np.maximum(probP[a:b], y[k, iP, BL:NW - BL])
            probS[a:b] = np.maximum(probS[a:b], y[k, iS, BL:NW - BL])
    valid_samples = len(good) * STRIDE
    picks = []
    for ph, pr, thr in (("P", probP, P_THR), ("S", probS, S_THR)):
        pk, prop = find_peaks(pr, height=thr, distance=100)
        for i, h in zip(pk, prop["peak_heights"]):
            picks.append((day0 + i / (FS * UP), ph, float(h)))
    return picks, min(1.0, valid_samples / N)


days = sorted(glob.glob(str(root / win / sta / "*.mseed")))
rows, upt = [], []
t0 = time.time()
for fn in days:
    dstr = Path(fn).stem
    day0 = obspy.UTCDateTime(dstr)
    tcut = obspy.UTCDateTime(dstr + "T23:46:00") if dstr.endswith("04-01") else None
    cache = outd / "days" / f"{win}_{sta}_{dstr}.parquet"
    if cache.exists():
        dfp = pd.read_parquet(cache)
        u = float(dfp.attrs.get("uptime", np.nan)) if "uptime" in dfp.attrs else float(dfp["uptime"].iloc[0])
        dfp = dfp[dfp.phase != "none"]
        picks = [(obspy.UTCDateTime(t.to_pydatetime()), ph, h) for t, ph, h in zip(dfp.time, dfp.phase, dfp.prob)]
    else:
        picks, u = run_day(fn, day0, tcut)
        cache.parent.mkdir(parents=True, exist_ok=True)
        dfp = pd.DataFrame([dict(time=pd.Timestamp(t.datetime), phase=ph, prob=h) for t, ph, h in picks]
                           or [dict(time=pd.Timestamp(dstr), phase="none", prob=0.0)])
        dfp["uptime"] = u
        dfp.to_parquet(cache)
    for t, ph, h in picks:
        rows.append(dict(station=sta, phase=ph, time=pd.Timestamp(t.datetime), prob=h))
    upt.append(dict(window=win, station=sta, day=dstr, uptime=u, n_P=sum(p[1] == "P" for p in picks),
                    n_S=sum(p[1] == "S" for p in picks)))
    print(win, sta, dstr, f"up={u:.2f}", len(picks), f"{time.time() - t0:.0f}s", flush=True)
pd.DataFrame(rows).to_parquet(outd / f"picks_{win}_{sta}.parquet")
pd.DataFrame(upt).to_csv(outd / f"uptime_{win}_{sta}.csv", index=False)
print("DONE", win, sta, len(rows), f"{time.time() - t0:.0f}s")
