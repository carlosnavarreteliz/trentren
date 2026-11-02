"""Second pass over the waveforms: Wood-Anderson amplitudes (magnitude proxy), PSGCX snippets for
repeating-family detection, and hourly 2-8 Hz envelope medians (tremor-like screening).

Usage: /opt/miniconda3/bin/python3 -I measure.py <mseed_root> <assoc_dir> <out_dir> <window>
Outputs in out_dir: amps_<win>.parquet (event_id, station, A_mm, R_km),
                    snip_<win>.npz (event_ids, PSGCX Z snippets 1-8 Hz, 20 Hz, P-1 s .. P+11 s),
                    env_<win>.parquet (station, hour, median envelope 2-8 Hz)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import obspy
from scipy.signal import butter, sosfiltfilt, hilbert

root, adir, odir, win = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
odir.mkdir(parents=True, exist_ok=True)
FS = 20.0
STA = {"PSGCX": (-19.59717, -70.12305, 629145000.0), "PB11": (-19.76096, -69.65582, 2516580000.0),
       "PATCX": (-20.82071, -70.15288, 629145000.0)}
VP, VS = 6.8, 3.9
SOS_WA = butter(4, 1.0, btype="high", fs=FS, output="sos")
SOS_SN = butter(4, [1.0, 8.0], btype="band", fs=FS, output="sos")
SOS_EN = butter(4, [2.0, 8.0], btype="band", fs=FS, output="sos")
# Wood-Anderson for velocity input: H(s) = G * s / (s^2 + 2 h w0 s + w0^2), G = 2080, T0 = 0.8 s, h = 0.7
W0, H, G = 2 * np.pi / 0.8, 0.7, 2080.0

ev = pd.read_parquet(adir / f"events_{win}.parquet")
ev = ev[ev.n_sta >= 3].copy()
asg = pd.read_parquet(adir / f"assign_{win}.parquet")
ev["day"] = ev.origin.dt.strftime("%Y-%m-%d")


def hyp_dist(lat, lon, dep, s):
    la, lo, _ = STA[s]
    dx = (lon - lo) * 111.19 * np.cos(np.radians(0.5 * (lat + la)))
    dy = (lat - la) * 111.19
    return np.sqrt(dx ** 2 + dy ** 2 + dep ** 2)


def wa_filter(v):
    """velocity (m/s) -> WA displacement (mm), frequency domain."""
    n = len(v); nf = 1 << int(np.ceil(np.log2(n)))
    f = np.fft.rfftfreq(nf, 1 / FS); s = 2j * np.pi * f
    Hs = G * s / (s ** 2 + 2 * H * W0 * s + W0 ** 2)
    return np.fft.irfft(np.fft.rfft(v, nf) * Hs, nf)[:n] * 1000.0


def load(fn, day0):
    n = int(86400 * FS); out = {}
    try:
        st = obspy.read(str(fn))
    except Exception:
        return out
    for tr in st:
        if tr.stats.sampling_rate == 40.0:
            tr.decimate(2)
    for c in "ZNE":
        arr = np.full(n, np.nan, np.float64)
        for tr in st.select(component=c):
            if tr.stats.sampling_rate != FS or len(tr.data) < 200:
                continue
            i0 = int(round((tr.stats.starttime - day0) * FS))
            d = tr.data.astype(np.float64); d -= d.mean()
            a, b = max(i0, 0), min(i0 + len(d), n)
            if b > a:
                arr[a:b] = d[a - i0:b - i0]
        out[c] = arr
    return out


amps, snips, snip_ids, env_rows = [], [], [], []
days = sorted({p.stem for s in STA for p in (root / win / s).glob("*.mseed")})
for dstr in days:
    day0 = obspy.UTCDateTime(dstr); t0 = pd.Timestamp(dstr)
    evd = ev[ev.day == dstr]
    for s, (la, lo, sens) in STA.items():
        fn = root / win / s / f"{dstr}.mseed"
        if not fn.exists():
            continue
        D = load(fn, day0)
        if not D:
            continue
        # hourly envelope medians (Z, 2-8 Hz)
        z = D["Z"]; good = ~np.isnan(z)
        zz = np.where(good, z, 0.0)
        env = np.abs(hilbert(sosfiltfilt(SOS_EN, zz))) / sens
        for h in range(24):
            sl = slice(int(h * 3600 * FS), int((h + 1) * 3600 * FS))
            g = good[sl]
            env_rows.append(dict(station=s, hour=t0 + pd.Timedelta(hours=h),
                                 env=float(np.median(env[sl][g])) if g.mean() > 0.9 else np.nan))
        if len(evd) == 0:
            continue
        hor = {}
        for c in "NE":
            x = D[c]; g = ~np.isnan(x)
            hor[c] = (np.where(g, wa_filter(sosfiltfilt(SOS_WA, np.where(g, x, 0.0)) / sens), np.nan))
        zsn = np.where(good, sosfiltfilt(SOS_SN, zz), np.nan)
        for _, e in evd.iterrows():
            R = hyp_dist(e.latitude, e.longitude, max(e.depth, 0), s)
            to = (e.origin - t0).total_seconds()
            a = int((to + R / VP - 1) * FS); b = int((to + R / VS + max(8.0, 0.5 * R / VS)) * FS)
            if a < 0 or b > len(z):
                continue
            pk = [np.nanmax(np.abs(hor[c][a:b])) for c in "NE" if np.isfinite(hor[c][a:b]).mean() > 0.9]
            if pk:
                amps.append(dict(event_id=e.event_id, station=s, A_mm=float(max(pk)), R_km=float(R)))
            if s == "PSGCX":
                pp = asg[(asg.event_id == e.event_id) & (asg.station == "PSGCX") & (asg.phase == "P")]
                tp = ((pd.to_datetime(pp.time.iloc[0], unit="s") - t0).total_seconds() if len(pp)
                      else to + R / VP)
                i0 = int(round((tp - 1.0) * FS)); i1 = i0 + int(12 * FS)
                if i0 >= 0 and i1 <= len(zsn):
                    w = zsn[i0:i1]
                    if np.isfinite(w).all():
                        snips.append(w.astype(np.float32)); snip_ids.append(e.event_id)
    print(win, dstr, len(evd), flush=True)

pd.DataFrame(amps).to_parquet(odir / f"amps_{win}.parquet")
pd.DataFrame(env_rows).to_parquet(odir / f"env_{win}.parquet")
np.savez_compressed(odir / f"snip_{win}.npz", ids=np.array(snip_ids), w=np.array(snips) if snips else np.zeros((0, 240)))
print("DONE", win, len(amps), len(snips))
