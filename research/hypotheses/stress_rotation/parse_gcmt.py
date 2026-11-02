"""Parse GCMT NDK files -> CSV of Chilean-margin moment tensors.

Usage: /opt/miniconda3/bin/python3 -I parse_gcmt.py <data_dir> <out_csv>
<data_dir> contains jan76_dec20.ndk, monthly/*.ndk, qcmt.ndk.
Monthly (final) solutions are used up to their last month; quick CMT only for later events.
Region kept: lat -48..-15, lon -80..-64 (wider than the harness region so edge bins have neighbours).
"""
import sys, glob, os
import numpy as np
import pandas as pd


def parse_ndk(path):
    with open(path, "r", errors="replace") as f:
        lines = [l.rstrip("\n") for l in f]
    out = []
    for i in range(0, len(lines) - 4, 5):
        l1, l2, l3, l4, l5 = lines[i:i + 5]
        try:
            date = l1[5:15].strip(); tim = l1[16:26].strip()
            hh, mm, ss = tim.split(":")
            ss = float(ss)
            t = pd.Timestamp(date.replace("/", "-")) + pd.Timedelta(hours=int(hh), minutes=int(mm), seconds=ss)
            name = l2[:16].strip()
            c = l3.split()
            # CENTROID: tshift err lat err lon err depth err ...
            tshift = float(c[1]); clat = float(c[3]); clon = float(c[5]); cdep = float(c[7])
            v = l4.split()
            ex = int(v[0])
            mrr, mtt, mpp, mrt, mrp, mtp = [float(v[k]) for k in (1, 3, 5, 7, 9, 11)]
            w = l5.split()
            ver = w[0]
            eT, plT, azT, eN, plN, azN, eP, plP, azP = [float(x) for x in w[1:10]]
            m0 = float(w[10])
            s1, d1, r1, s2, d2, r2 = [float(x) for x in w[11:17]]
        except Exception as e:  # malformed block
            print("skip block", path, i, e, file=sys.stderr)
            continue
        out.append(dict(time=t, name=name, lat=clat, lon=clon, depth=cdep, tshift=tshift, exp=ex,
                        mrr=mrr, mtt=mtt, mpp=mpp, mrt=mrt, mrp=mrp, mtp=mtp,
                        eT=eT, plT=plT, azT=azT, eN=eN, plN=plN, azN=azN, eP=eP, plP=plP, azP=azP,
                        m0=m0 * 10.0 ** ex, strike1=s1, dip1=d1, rake1=r1, strike2=s2, dip2=d2, rake2=r2,
                        src=os.path.basename(path)))
    return pd.DataFrame(out)


def main(data_dir, out_csv):
    frames = [parse_ndk(os.path.join(data_dir, "jan76_dec20.ndk"))]
    for p in sorted(glob.glob(os.path.join(data_dir, "monthly", "*.ndk"))):
        frames.append(parse_ndk(p))
    final = pd.concat(frames, ignore_index=True)
    last_final = final.time.max()
    q = parse_ndk(os.path.join(data_dir, "qcmt.ndk"))
    q = q[q.time > last_final + pd.Timedelta(days=1)]
    q["quick"] = True
    final["quick"] = False
    df = pd.concat([final, q], ignore_index=True)
    # the 'time' in line 1 is the reference hypocentre time; centroid time = time + tshift
    df["mw"] = (2.0 / 3.0) * (np.log10(df.m0) - 16.1)
    reg = df[(df.lat >= -48) & (df.lat <= -15) & (df.lon >= -80) & (df.lon <= -64)].copy()
    reg = reg.sort_values("time").drop_duplicates("name").reset_index(drop=True)
    reg.to_csv(out_csv, index=False)
    print("global events:", len(df), "final up to", last_final, "| regional:", len(reg),
          "| quick regional:", int(reg.quick.sum()))
    print(reg.groupby(reg.time.dt.year // 5 * 5).size().to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
